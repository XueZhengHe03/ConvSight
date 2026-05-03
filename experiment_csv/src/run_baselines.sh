#!/bin/bash

# run_baselines.sh - 最终修复版
echo "🚀 开始批量评估所有模型..."
echo

declare -A RESULTS

declare -A EVAL_SCRIPTS=(
    ["TS2Vec"]="evaluate_ts2vec.py"
    ["TimesNet"]="evaluate_timesnet.py"
    ["iTransformer"]="evaluate_iTransformer.py"
    ["Non-stationary Transformer"]="evaluate_Nonstationary_Transformer.py"
)

for model in "${!EVAL_SCRIPTS[@]}"; do
    script="${EVAL_SCRIPTS[$model]}"
    if [ ! -f "$script" ]; then
        echo "⚠️  $model 脚本不存在: $script"
        RESULTS["$model"]="N/A,N/A,N/A"
        continue
    fi

    echo "🔍 正在评估 $model ..."
    output=$(python "$script" 2>/dev/null)

    # 修复：使用 [[:space:]]* 匹配任意空白
    acc=$(echo "$output" | grep "Overall Accuracy" | sed -E 's/.*:[[:space:]]*([0-9.]+).*/\1/')
    conv=$(echo "$output" | grep "Converged Recall" | sed -E 's/.*:[[:space:]]*([0-9.]+).*/\1/')
    unconv=$(echo "$output" | grep "Unconverged Recall" | sed -E 's/.*:[[:space:]]*([0-9.]+).*/\1/')

    if [[ "$acc" =~ ^[0-9.]+$ && "$conv" =~ ^[0-9.]+$ && "$unconv" =~ ^[0-9.]+$ ]]; then
        RESULTS["$model"]="$acc,$conv,$unconv"
        echo "✅ $model: Acc=$acc, ConvRecall=$conv, UnconvRecall=$unconv"
    else
        echo "❌ $model: 无法提取指标"
        echo "   调试行:"
        echo "$output" | grep -E "(Accuracy|Recall)" | sed 's/^/      /'
        RESULTS["$model"]="N/A,N/A,N/A"
    fi
    echo
done

echo "📊 模型性能对比汇总:"
echo
printf "%-25s %-15s %-20s %s\n" "Model" "Overall Acc" "Converged Recall" "Unconverged Recall"
printf "%-25s %-15s %-20s %s\n" "-------------------------" "---------------" "--------------------" "----------------------"

for model in "TS2Vec" "TimesNet" "iTransformer" "Non-stationary Transformer"; do
    if [[ -n "${RESULTS[$model]}" ]]; then
        IFS=',' read -r acc conv unconv <<< "${RESULTS[$model]}"
        printf "%-25s %-15s %-20s %s\n" "$model" "$acc" "$conv" "$unconv"
    else
        printf "%-25s %-15s %-20s %s\n" "$model" "N/A" "N/A" "N/A"
    fi
done

echo
echo "✅ 所有评估完成！"