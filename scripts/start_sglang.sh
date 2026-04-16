#!/bin/bash

MODEL_DIR="$HOME/.cache/huggingface/hub/models--pengg--EvoDriveVLA-drone/snapshots/1a6cfd435a7f314e3a487e63be8f0b37c567c463"
# MODEL_DIR="$HOME/.cache/huggingface/hub/EvoDriveVLA-drone-student-only"
CONFIG_PATCH="/tmp/sglang_config.json"
PREPROCESSOR_PATCH="/tmp/sglang_preprocessor_config.json"
PATCH_SCRIPT="/tmp/patch_sglang.py"

# Step 1: 修复 config.json（去掉 KD 后缀，sglang 不认识 Qwen2_5_VLForConditionalGeneration_KD）
cp "$MODEL_DIR/config.json" "$CONFIG_PATCH"
sed -i 's/Qwen2_5_VLForConditionalGeneration_KD/Qwen2_5_VLForConditionalGeneration/' "$CONFIG_PATCH"

# Step 2: 还原 preprocessor_config.json（使用原始版本，容器内 transformers 5.3.0 支持）
cp "$MODEL_DIR/preprocessor_config.json" "$PREPROCESSOR_PATCH"

# Step 3: 生成 patch 脚本（跳过 KD teacher 权重，避免 KeyError/ValueError）
cat > "$PATCH_SCRIPT" << 'EOF'
import re

# Patch 1: utils.py - 遇到不认识的模块名时跳过而不是报错
path1 = '/sgl-workspace/sglang/python/sglang/srt/models/utils.py'
with open(path1, 'r') as f:
    code = f.read()

old1 = (
    '            raise ValueError(\n'
    '                f"No module or parameter named {prefix!r} in {self.module._get_name()}."\n'
    '            )'
)
new1 = (
    '            # skip unknown weights (e.g. KD teacher weights)\n'
    '            continue'
)

if old1 in code:
    code = code.replace(old1, new1)
    with open(path1, 'w') as f:
        f.write(code)
    print("Patched utils.py successfully.")
else:
    print("utils.py already patched or pattern not found, skipping.")

# Patch 2: qwen2_5_vl.py - 跳过不在 params_dict 里的 key（如 encoder_teacher.*）
path2 = '/sgl-workspace/sglang/python/sglang/srt/models/qwen2_5_vl.py'
with open(path2, 'r') as f:
    code = f.read()

old2 = (
    '                param = params_dict[name]\n'
    '                weight_loader = param.weight_loader\n'
    '                weight_loader(param, loaded_weight, shard_id)\n'
    '                break'
)
new2 = (
    '                if name not in params_dict:\n'
    '                    continue\n'
    '                param = params_dict[name]\n'
    '                weight_loader = param.weight_loader\n'
    '                weight_loader(param, loaded_weight, shard_id)\n'
    '                break'
)

if old2 in code:
    code = code.replace(old2, new2)
    with open(path2, 'w') as f:
        f.write(code)
    print("Patched qwen2_5_vl.py successfully.")
else:
    print("qwen2_5_vl.py already patched or pattern not found, skipping.")
EOF

# Step 4: 启动容器
docker run --gpus all --shm-size 32g -p 30000:30000 \
  -v "$HOME/.cache/huggingface/hub/models--pengg--EvoDriveVLA-drone/:/model-root" \
  -v "$CONFIG_PATCH:/model-root/snapshots/1a6cfd435a7f314e3a487e63be8f0b37c567c463/config.json" \
  -v "$PREPROCESSOR_PATCH:/model-root/snapshots/1a6cfd435a7f314e3a487e63be8f0b37c567c463/preprocessor_config.json" \
  -v "$PATCH_SCRIPT:/patch_sglang.py" \
  lmsysorg/sglang:latest \
  bash -c "python3 /patch_sglang.py && python3 -m sglang.launch_server \
    --model-path /model-root/snapshots/1a6cfd435a7f314e3a487e63be8f0b37c567c463 \
    --host 0.0.0.0 --port 30000 --trust-remote-code"
