import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def get_all_model_data():
    runs_dir = r'e:\PycharmProjects\人脸识别\runs\detect'
    model_data = {}
    summary_data = []
    
    # 只比较这四个模型
    target_models = {
        'face_emotion_v8m': 'YOLOv8m',
        'face_emotion_v11n': 'YOLOv11n',
        'face_emotion_v8s': 'YOLOv8s',
        'face_emotion_v8n': 'YOLOv8n'
    }
    
    # 颜色列表
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    
    # 遍历目标模型文件夹
    color_idx = 0
    for folder, model_name in target_models.items():
        folder_path = os.path.join(runs_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"警告：文件夹 {folder} 不存在")
            continue
        
        # 查找 CSV 文件
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not csv_files:
            print(f"警告：{folder} 中没有 CSV 文件")
            continue
        
        # 读取 CSV 文件
        csv_path = os.path.join(folder_path, csv_files[0])
        try:
            df = pd.read_csv(csv_path)
            
            # 获取统计数据
            metric_mappings = {
                'metrics/precision(B)': 'precision',
                'metrics/recall(B)': 'recall', 
                'metrics/mAP50(B)': 'mAP_0.5',
                'metrics/mAP50-95(B)': 'mAP_0.5-0.95'
            }
            
            for csv_metric, display_metric in metric_mappings.items():
                if csv_metric not in df.columns:
                    continue
                    
                values = df[csv_metric].values
                min_val = np.min(values)
                max_val = np.max(values)
                start_val = values[0]
                end_val = values[-1]
                delta_val = end_val - start_val
                delta_pct = (delta_val / start_val * 100) if start_val != 0 else 0
                start_step = 0
                end_step = len(values) - 1
                
                summary_data.append({
                    '模型名称': model_name,
                    '指标': display_metric,
                    'Min': min_val,
                    'Max': max_val,
                    'Start Value': start_val,
                    'End Value': end_val,
                    'ΔValue': delta_val,
                    'Δ%': delta_pct,
                    'Start Step': start_step,
                    'End Step': end_step
                })
            
            # 保存完整数据用于绘图
            model_data[model_name] = {
                'data': df,
                'color': colors[color_idx]
            }
            color_idx += 1
            
        except Exception as e:
            print(f"读取 {folder} 失败：{e}")
            continue
    
    return model_data, pd.DataFrame(summary_data)

def smooth_curve(values, window=5):
    """使用滑动平均平滑曲线，保持首尾数据不变"""
    if len(values) < window:
        return values
    
    smoothed = values.copy()
    half_window = window // 2
    
    # 只对中间部分进行平滑，首尾保持原值
    for i in range(half_window, len(values) - half_window):
        smoothed[i] = np.mean(values[i - half_window:i + half_window + 1])
    
    return smoothed

def plot_wandb_style_curves(model_data, summary_df, metric_name='metrics/mAP50(B)', display_name='mAP_0.5'):
    # 创建图表，使用 GridSpec 来布局曲线和表格
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 0.5])
    
    # 上方绘制曲线
    ax_curve = fig.add_subplot(gs[0])
    
    # 找到所有模型中最长的 epoch 数，用于设置 x 轴范围
    max_epochs = max([len(model_info['data']) for model_info in model_data.values()])
    
    # 绘制每个模型的曲线（使用平滑处理）
    for model_name, model_info in model_data.items():
        df = model_info['data']
        color = model_info['color']
        
        if metric_name in df.columns:
            epochs = df.index + 1  # 从 1 开始
            values = df[metric_name].values
            
            # 使用滑动平均平滑曲线，窗口大小为 5
            smoothed_values = smooth_curve(values, window=5)
            
            # 绘制平滑后的曲线（只画到该模型实际的 epoch 数）
            ax_curve.plot(epochs[:len(smoothed_values)], smoothed_values, linewidth=2.5, color=color, label=model_name, alpha=0.85)
    
    # 设置曲线图表
    ax_curve.set_title(f'metrics/{display_name}', fontsize=16, fontweight='bold', pad=20)
    ax_curve.set_ylabel('数值', fontsize=12)
    ax_curve.set_xlabel('Epoch', fontsize=12)
    ax_curve.grid(True, alpha=0.3, linestyle='--')
    ax_curve.legend(loc='lower right', fontsize=10, bbox_to_anchor=(1, 0))
    
    # 设置 x 轴范围为所有模型的最大 epoch
    ax_curve.set_xlim(0.5, max_epochs + 0.5)
    
    # 添加起始和结束标记
    if model_data:
        first_model = list(model_data.keys())[0]
        df_first = model_data[first_model]['data']
        max_epoch = len(df_first)
        
        # 添加起始和结束的虚线
        ax_curve.axvline(x=0.5, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
        ax_curve.axvline(x=max_epoch, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
        
        # 添加文字标注
        ax_curve.text(0.5, ax_curve.get_ylim()[0], ' 0×', 
                     ha='left', va='bottom', fontsize=9, fontweight='bold')
        ax_curve.text(max_epoch, ax_curve.get_ylim()[0], f' {max_epoch}×', 
                     ha='right', va='bottom', fontsize=9, fontweight='bold')
    
    # 中间绘制统计表
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('tight')
    ax_table.axis('off')
    
    # 筛选当前指标的数据
    table_df = summary_df[summary_df['指标'] == display_name].copy()
    table_df = table_df[['模型名称', 'Min', 'Max', 'Start Value', 'End Value', 'ΔValue', 'Δ%', 'Start Step', 'End Step']]
    
    # 格式化数据
    for col in ['Min', 'Max', 'Start Value', 'End Value', 'ΔValue']:
        table_df[col] = table_df[col].apply(lambda x: f"{x:.4f}")
    table_df['Δ%'] = table_df['Δ%'].apply(lambda x: f"{x:.0f}%")
    
    # 创建表格
    table = ax_table.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.2, 0.08, 0.08, 0.1, 0.1, 0.1, 0.08, 0.08, 0.08]
    )
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # 设置表头样式
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#333333')
            cell.set_text_props(weight='bold', color='white')
        else:
            if i % 2 == 0:
                cell.set_facecolor('#f8f8f8')
    
    plt.tight_layout()
    plt.savefig(f'{display_name}_训练曲线对比.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_comparison_table(df):
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 格式化数据，保留4位小数
    df_formatted = df.copy()
    df_formatted['Precision'] = df_formatted['Precision'].apply(lambda x: f"{x:.4f}")
    df_formatted['Recall'] = df_formatted['Recall'].apply(lambda x: f"{x:.4f}")
    df_formatted['mAP_0.5'] = df_formatted['mAP_0.5'].apply(lambda x: f"{x:.4f}")
    df_formatted['mAP_0.5:0.95'] = df_formatted['mAP_0.5:0.95'].apply(lambda x: f"{x:.4f}")
    
    # 创建表格
    table = ax.table(
        cellText=df_formatted.values,
        colLabels=df_formatted.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.15, 0.15, 0.15, 0.15]
    )
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # 设置表头样式
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        else:
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
    
    plt.title('人脸表情识别模型性能对比表', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('模型性能对比表.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_bar_charts(df):
    # 创建四个柱状图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('人脸表情识别模型性能对比', fontsize=20, fontweight='bold')
    
    metrics = [
        ('Precision', '精确率 (Precision)', '#4CAF50'),
        ('Recall', '召回率 (Recall)', '#2196F3'),
        ('mAP_0.5', 'mAP@0.5', '#FF9800'),
        ('mAP_0.5:0.95', 'mAP@0.5:0.95', '#9C27B0')
    ]
    
    for i, (metric, title, color) in enumerate(metrics):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # 按指标排序
        df_sorted = df.sort_values(by=metric, ascending=False)
        
        # 绘制柱状图
        bars = ax.bar(df_sorted['模型名称'], df_sorted[metric], color=color, alpha=0.7)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('数值', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('模型性能对比柱状图.png', dpi=300, bbox_inches='tight')
    plt.show()

def get_model_results():
    runs_dir = r'e:\PycharmProjects\人脸识别\runs\detect'
    results = []
    
    # 只比较这四个模型
    target_models = {
        'face_emotion_v8m': 'YOLOv8m',
        'face_emotion_v11n': 'YOLOv11n',
        'face_emotion_v8s': 'YOLOv8s',
        'face_emotion_v8n': 'YOLOv8n'
    }
    
    # 遍历目标模型文件夹
    for folder, model_name in target_models.items():
        folder_path = os.path.join(runs_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"警告：文件夹 {folder} 不存在")
            continue
        
        # 查找 CSV 文件
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not csv_files:
            print(f"警告：{folder} 中没有 CSV 文件")
            continue
        
        # 读取 CSV 文件
        csv_path = os.path.join(folder_path, csv_files[0])
        try:
            df = pd.read_csv(csv_path)
            
            # 获取最后一行（最好的 epoch）
            last_row = df.iloc[-1]
            
            # 提取需要的数据
            precision = last_row['metrics/precision(B)']
            recall = last_row['metrics/recall(B)']
            map50 = last_row['metrics/mAP50(B)']
            map50_95 = last_row['metrics/mAP50-95(B)']
            
            results.append({
                '模型名称': model_name,
                'Precision': precision,
                'Recall': recall,
                'mAP_0.5': map50,
                'mAP_0.5:0.95': map50_95
            })
        except Exception as e:
            print(f"读取 {folder} 失败：{e}")
            continue
    
    return pd.DataFrame(results)

if __name__ == '__main__':
    # 获取所有模型结果
    df_summary = get_model_results()
    
    if df_summary.empty:
        print("没有找到模型结果！")
    else:
        print("找到以下模型结果：")
        print(df_summary)
        print("\n")
        
        # 保存CSV表格
        df_summary.to_csv('模型性能对比.csv', index=False, encoding='utf-8-sig')
        print("已保存：模型性能对比.csv")
        
        # 获取完整数据用于绘制训练曲线
        model_data, summary_df = get_all_model_data()
        
        if model_data:
            # 绘制mAP_0.5的曲线对比图
            plot_wandb_style_curves(model_data, summary_df, 'metrics/mAP50(B)', 'mAP_0.5')
            
            # 绘制mAP_0.5:0.95的曲线对比图
            plot_wandb_style_curves(model_data, summary_df, 'metrics/mAP50-95(B)', 'mAP_0.5-0.95')
            
            # 绘制Precision的曲线对比图
            plot_wandb_style_curves(model_data, summary_df, 'metrics/precision(B)', 'precision')
            
            # 绘制Recall的曲线对比图
            plot_wandb_style_curves(model_data, summary_df, 'metrics/recall(B)', 'recall')
        
        # 生成对比表
        plot_comparison_table(df_summary)
        
        # 生成柱状图
        plot_bar_charts(df_summary)
        
        print("\n所有图表已生成完成！")