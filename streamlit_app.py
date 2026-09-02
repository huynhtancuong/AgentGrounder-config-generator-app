import streamlit as st
import yaml

st.set_page_config(page_title="Experiment Config Generator", layout="wide")
st.title("🛠️ Experiment Profile Generator")

col_input, col_preview = st.columns([2, 2])

with col_input:
    # -----------------------------
    # 1. EXPERIMENT & DATASET
    # -----------------------------
    st.subheader("1. Input Settings")
    dataset = st.selectbox("Dataset", ["scanrefer", "nr3d"])
    scan_type = st.selectbox("Scan Type", ["mesh", "pointcloud", "gaussian"])
    subset_name = st.selectbox("Subset Name", ["validation_set_strats_10query", "validation_set_strats_60query", "validation_set_strats_120query", "validation_set_strats_1000query"])
    bounding_box_source = st.selectbox("Bounding Box Source", ["pred", "gt", "no bbox"])


    base_data_path = f"/preprocessed_data/{dataset}"

    lang_dir = f"{base_data_path}/query/{subset_name}"
    data_dir = f"/data/{scan_type}_scans/"

    pred_bbox_dir = f"{base_data_path}/object_lookup_table/{bounding_box_source}"

    if bounding_box_source == "pred":
        if st.checkbox("Use Extended Object Lookup Table", value=False):
            pred_bbox_dir = f"{base_data_path}/extended_object_lookup_table/"
    if bounding_box_source == "no bbox":
        st.info("3D Segmentor will be used for generating bounding boxes.")
        pred_bbox_dir = ""

    rev_number = st.number_input("Revision Number (to be appended to output directory)", value=1, step=1)

    output_dir = f"/data/output/{dataset}/{scan_type}/{subset_name}/rev_{rev_number}/"
    
    # -----------------------------
    # 2. MODEL & SERVER CONFIG
    # -----------------------------
    st.subheader("2. VLM Model Configuration")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        model_name = st.selectbox("Model Name", ["gemma4:31b",
                                               "qwen3.8:27b",
                                               "qwen3.5:35b",
                                               "qwen3-vl:32b-instruct",
                                               "qwen2.4vl:72b"])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.8, step=0.05)
    with col_m2:
        base_url = st.text_input("Base URL", "http://192.168.0.221:11435,http://192.168.0.221:11436")
        num_ctx = st.number_input("Context Window (num_ctx)", value=16384, step=1024)

    # -----------------------------
    # 3. LANDMARK PARSER & WEIGHTS
    # -----------------------------
    st.subheader("3. Landmark Parser Settings")
    extr_mode = st.selectbox(
                    "Extraction Mode",
                    ["embed", "dummy", "nlp"]
                )
    top_k = 0
    score_threshold = 0.0
    vectorstore_dir = f"{base_data_path}/chromadb_data/mask3d_labels_{bounding_box_source}"

    if extr_mode == "embed":
        with st.expander("Embedding Mode Settings", expanded=True):
            default_top_k = 40
            top_k = st.number_input("Top K", value=default_top_k, step=5)
            score_threshold = st.slider("Score Threshold", 0.0, 1.0, 0.8, step=0.05)
            if bounding_box_source == "pred":
                if st.checkbox("Use OpenVocab Description", value=False):
                    vectorstore_dir = f"{base_data_path}/chromadb_data/openvocab-description/"
                    st.info("OpenVocab Description will be used for embedding.")
                else:
                    st.info("Labels will be used for embedding.")

    estim_mode = st.selectbox("Estimation Mode", ["fill","predict"])
    if estim_mode == "predict":
        st.info("3D Segmentor will be used for prediction.")


    # device = st.selectbox("Compute Device", ["cuda", "cuda:0", "cuda:1", "cpu"])

    device = "cuda"

    # -----------------------------
    # 4. Agent Settings
    # -----------------------------

    st.subheader("4. Agent Settings")
    objects_info_strategy = st.selectbox("Input prompt strategy", ["full", "labels_only"])
    if objects_info_strategy == "labels_only":
        st.info("Input prompt will only include labels of objects in the scene.")
    else:
        st.info("Input prompt will include full information of all objects in the scene.")

    st.write("**Agent Tools:**")

    # Tool Metadata Registry with declarative dependencies
    TOOL_REGISTRY = [
        {
            "id": "write_plan",
            "description": "Plan step-by-step reasoning actions",
        },
        {
            "id": "see_image",
            "description": "View overall scene visualization",
            "requirements": [
                {
                    "check": lambda ctx: ctx.get("bounding_box_source") != "no bbox",
                    "message": "Requires bounding box source to be not 'no bbox'.",
                }
            ],
        },
        {
            "id": "see_image_from_specific_view",
            "description": "View object from specific camera angle",
            "requirements": [
                {
                    "check": lambda ctx: ctx.get("bounding_box_source") != "no bbox",
                    "message": "Requires bounding box source to be not 'no bbox'.",
                }
            ],
        },
        {
            "id": "get_object_info",
            "description": "Get detailed bounding box and object information",
            "requirements": [
                {
                    "check": lambda ctx: ctx.get("bounding_box_source") != "no bbox",
                    "message": "Requires bounding box source to be not 'no bbox'.",
                }
            ],
        },
        {
            "id": "calculate_distance",
            "description": "Calculate Euclidean distance between objects",
            "requirements": [
                {
                    "check": lambda ctx: ctx.get("bounding_box_source") != "no bbox",
                    "message": "Requires bounding box source to be not 'no bbox'.",
                }
            ],
        },
        {
            "id": "find_new_object_in_scene",
            "description": "Segment and discover new objects in 3D scene",
            "requirements": [
                {
                    "check": lambda ctx: ctx.get("estim_mode") == "predict",
                    "message": "Requires Estimation Mode = 'predict'.",
                }
            ],
        },
    ]

    # Evaluation context for tool requirements
    context = {
        "estim_mode": estim_mode,
        "bounding_box_source": bounding_box_source,
        "scan_type": scan_type,
        "dataset": dataset,
    }

    col_t1, col_t2 = st.columns(2)
    selected_tools = []
    for i, tool_meta in enumerate(TOOL_REGISTRY):
        target_col = col_t1 if i % 2 == 0 else col_t2
        tool_id = tool_meta["id"]

        # Validate all requirements for this tool
        unmet_requirements = [
            req["message"]
            for req in tool_meta.get("requirements", [])
            if not req["check"](context)
        ]
        is_disabled = len(unmet_requirements) > 0
        default_val = False if is_disabled else True

        with target_col:
            checked = st.checkbox(
                tool_id,
                value=default_val,
                disabled=is_disabled,
                help=tool_meta.get("description"),
            )
            if checked and not is_disabled:
                selected_tools.append(tool_id)
            if is_disabled:
                for req_msg in unmet_requirements:
                    st.info(f"`{tool_id}`: {req_msg}")

# -----------------------------
# 5. Generate Configuration Dictionary
# -----------------------------
config_data = {
    "model": {
        "api_key": "your_openai_api_key_here",
        "base_url": base_url,
        "name": model_name,
        "temperature": temperature,
        "num_ctx": num_ctx
    },
    "prompts": {
        "inference_prompt": "!include inference_prompt.yaml",
        "ov_caption_prompt": "!include ov_caption_prompt.yaml"
    },
    "rendering": {
        "use_image": True
    },
    "landmark_parser": {
        "extr_mode": extr_mode,
        "estim_mode": estim_mode,
        "top_k": top_k,
        "score_threshold": score_threshold,
        **({
            "segmentor3d": {
                "rex_folder_path": "/home/docker_user/AgentGrounder/weights/Rex-Omni",
                "sam3_bpe_path": "/home/docker_user/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                "sam3_weights_path": "/home/docker_user/AgentGrounder/weights/sam3/sam3.pt",
                "save_dir_base_subpath": "projection_img_aug/",
                "fixed_label": "[object]",
                "rex_device": device if "cuda" in device else "cpu",
                "sam3_device": device if "cuda" in device else "cpu"
            }
        } if estim_mode == "predict" else {}),
        "embedding_model": {
            "name": "embeddinggemma:latest"
        }
    },
    "agent": {
        "max_attempts": 5,
        "objects_info_strategy": objects_info_strategy,
        "prompt": "!include inference_prompt_agent.yaml",
        "tools": selected_tools
    },
    "renderer": {
        "device": device if "cuda" in device else "cpu",
        "image_size": 680,
        "filter_ceiling": True
    },
    "experiment": {
        "dataset": dataset,
        "split": "test",
        "scan_type": scan_type,
        "device": device,
        "output_dir": output_dir,
        "data": {
            "language_annotation_dir": lang_dir,
            "gt_bbox_dir": f"{base_data_path}/object_lookup_table/gt",
            "pred_bbox_dir": pred_bbox_dir,
            "data_dir": data_dir,
            "val_file": "/root/Qwen2-VL/AgentGrounder/data/scannet/scannetv2_val.txt",
            "open_vocab_captions_dir": f"{base_data_path}/open_vocab_captions_data/open_vocab_captions_v2",
            "vectorstore_dir": vectorstore_dir,
            "multi_view_projection_dir": f"{base_data_path}/multi_view_imgs_data/multi-view-projection-imgs_v3"
        }
    }
}

# Convert the config dictionary to a YAML string (Preserve the string directive format for !include)
yaml_string = yaml.dump(config_data, sort_keys=False, default_flow_style=False)
yaml_string = yaml_string.replace("'!include", "!include").replace("'", "")

# -----------------------------
# 6. PREVIEW & DOWNLOAD
# -----------------------------
with col_preview:
    st.subheader("📄 Config YAML Preview")
    st.code(yaml_string, language="yaml")
    
    st.download_button(
        label="💾 Download YAML Config",
        data=yaml_string,
        file_name=f"{dataset}_{scan_type}.yaml",
        mime="text/yaml"
    )