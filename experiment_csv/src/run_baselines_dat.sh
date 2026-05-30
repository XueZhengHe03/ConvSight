#!/bin/bash

# run_baselines_dat.sh - 新数据集批量评估
echo "🚀 开始批量评估所有模型 (dat dataset)..."
echo

declare -A RESULTS

declare -A EVAL_SCRIPTS=(
    ["TS2Vec"]="evaluate_ts2vec_dat.py"
    ["TimesNet"]="evaluate_timesnet_dat.py"
    ["iTransformer"]="evaluate_iTransformer_dat.py"
    ["Non-stationary Transformer"]="evaluate_Nonstationary_Transformer_dat.py"
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

echo "📊 模型性能对比汇总 (dat dataset):"
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
