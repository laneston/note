import akshare as ak
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import os
from datetime import datetime

def get_all_stock_codes():
    """
    获取所有A股股票代码列表
    """
    print("📋 获取A股股票列表...")
    try:
        # 获取所有A股股票代码
        stock_info_a_code_name = ak.stock_info_a_code_name()
        # 过滤掉北交所股票（以8开头），如果需要包含北交所可以去掉这个过滤
        stock_list = stock_info_a_code_name[~stock_info_a_code_name['code'].str.startswith('8')]
        print(f"✅ 获取到 {len(stock_list)} 只A股股票")
        return stock_list
    except Exception as e:
        print(f"❌ 获取股票列表失败: {e}")
        return None

def get_stock_pe_pb_batch(stock_codes, batch_size=100, delay=1):
    """
    批量获取股票的市盈率和市净率数据
    """
    all_data = []
    failed_codes = []
    
    total_batches = (len(stock_codes) + batch_size - 1) // batch_size
    
    print(f"📊 开始批量获取市盈率和市净率数据，共 {total_batches} 批次...")
    
    for i in tqdm(range(0, len(stock_codes), batch_size), desc="处理进度"):
        batch_codes = stock_codes[i:i + batch_size]
        batch_data = []
        
        for code in batch_codes:
            try:
                # 获取个股实时行情，包含市盈率和市净率
                stock_data = ak.stock_individual_info_em(symbol=code)
                
                # 提取需要的字段
                stock_dict = {
                    'code': code,
                    'name': get_stock_name(code, stock_codes)
                }
                
                # 从返回的数据中提取市盈率和市净率
                for _, row in stock_data.iterrows():
                    if row['item'] == '市盈率动':
                        stock_dict['pe_ratio'] = parse_ratio(row['value'])
                    elif row['item'] == '市净率':
                        stock_dict['pb_ratio'] = parse_ratio(row['value'])
                    elif row['item'] == '总市值':
                        stock_dict['market_cap'] = parse_market_cap(row['value'])
                    elif row['item'] == '流通市值':
                        stock_dict['circulating_market_cap'] = parse_market_cap(row['value'])
                
                # 如果成功获取到数据，添加到批次数据中
                if 'pe_ratio' in stock_dict or 'pb_ratio' in stock_dict:
                    batch_data.append(stock_dict)
                else:
                    failed_codes.append(code)
                    
            except Exception as e:
                failed_codes.append(code)
                continue
        
        all_data.extend(batch_data)
        
        # 添加延迟，避免请求过于频繁
        if i + batch_size < len(stock_codes):
            time.sleep(delay)
    
    print(f"✅ 成功获取 {len(all_data)} 只股票的数据")
    if failed_codes:
        print(f"⚠️  有 {len(failed_codes)} 只股票数据获取失败")
    
    return pd.DataFrame(all_data), failed_codes

def get_stock_pe_pb_alternative():
    """
    使用替代方法获取市盈率和市净率（更快速但可能数据不全）
    """
    print("🚀 使用快速方法获取市盈率市净率数据...")
    try:
        # 方法1：使用实时行情数据
        stock_zh_a_spot_df = ak.stock_zh_a_spot()
        
        # 选择需要的列并重命名
        columns_needed = ['代码', '名称', '市盈率-动态', '市净率', '总市值', '流通市值']
        available_columns = [col for col in columns_needed if col in stock_zh_a_spot_df.columns]
        
        data = stock_zh_a_spot_df[available_columns].copy()
        
        # 重命名列
        column_mapping = {
            '代码': 'code',
            '名称': 'name', 
            '市盈率-动态': 'pe_ratio',
            '市净率': 'pb_ratio',
            '总市值': 'market_cap',
            '流通市值': 'circulating_market_cap'
        }
        
        data.rename(columns=column_mapping, inplace=True)
        
        # 处理市盈率和市净率数据
        data['pe_ratio'] = data['pe_ratio'].apply(parse_ratio)
        data['pb_ratio'] = data['pb_ratio'].apply(parse_ratio)
        
        # 过滤掉没有市盈率或市净率的数据
        data = data[(data['pe_ratio'].notna()) | (data['pb_ratio'].notna())]
        
        print(f"✅ 快速方法获取到 {len(data)} 只股票的数据")
        return data, []
        
    except Exception as e:
        print(f"❌ 快速方法获取数据失败: {e}")
        return pd.DataFrame(), []

def parse_ratio(value):
    """
    解析比率数据，处理特殊值如'--', '-', '0.00'等
    """
    if pd.isna(value) or value in ['--', '-', '', '0.00', None]:
        return np.nan
    
    try:
        # 尝试转换为浮点数
        return float(str(value).replace(',', ''))
    except (ValueError, TypeError):
        return np.nan

def parse_market_cap(value):
    """
    解析市值数据，处理单位（亿、万）
    """
    if pd.isna(value) or value in ['--', '-', '', None]:
        return np.nan
    
    try:
        str_value = str(value).replace(',', '')
        if '亿' in str_value:
            return float(str_value.replace('亿', '')) * 100000000
        elif '万' in str_value:
            return float(str_value.replace('万', '')) * 10000
        else:
            return float(str_value)
    except (ValueError, TypeError):
        return np.nan

def get_stock_name(code, stock_list_df):
    """
    根据股票代码获取股票名称
    """
    try:
        name = stock_list_df[stock_list_df['code'] == code]['name'].values[0]
        return name
    except:
        return code

def analyze_and_sort_data(df):
    """
    分析数据并进行排序
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # 创建副本避免修改原数据
    df_pe = df.copy()
    df_pb = df.copy()
    
    # 过滤掉无效值（负值或特别大的异常值）
    df_pe = df_pe[(df_pe['pe_ratio'] > 0) & (df_pe['pe_ratio'] < 10000)]
    df_pb = df_pb[(df_pb['pb_ratio'] > 0) & (df_pb['pb_ratio'] < 100)]
    
    print(f"\n📊 数据分析:")
    print(f"   有效市盈率数据: {len(df_pe)} 只股票")
    print(f"   有效市净率数据: {len(df_pb)} 只股票")
    
    # 按市盈率排序（从小到大，市盈率越低越好）
    df_pe_sorted = df_pe.sort_values('pe_ratio', ascending=True).reset_index(drop=True)
    df_pe_sorted['pe_rank'] = df_pe_sorted.index + 1
    
    # 按市净率排序（从小到大，市净率越低越好）
    df_pb_sorted = df_pb.sort_values('pb_ratio', ascending=True).reset_index(drop=True)
    df_pb_sorted['pb_rank'] = df_pb_sorted.index + 1
    
    # 添加分位数信息
    df_pe_sorted['pe_percentile'] = df_pe_sorted['pe_rank'] / len(df_pe_sorted)
    df_pb_sorted['pb_percentile'] = df_pb_sorted['pb_rank'] / len(df_pb_sorted)
    
    return df_pe_sorted, df_pb_sorted

def generate_summary_report(pe_df, pb_df):
    """
    生成数据摘要报告
    """
    print("\n" + "="*80)
    print("📈 A股市盈率市净率分析报告")
    print("="*80)
    
    if not pe_df.empty:
        print(f"\n📊 市盈率(PE)统计:")
        print(f"   最低PE: {pe_df['pe_ratio'].min():.2f} - {pe_df.iloc[0]['name']}({pe_df.iloc[0]['code']})")
        print(f"   最高PE: {pe_df['pe_ratio'].max():.2f} - {pe_df.iloc[-1]['name']}({pe_df.iloc[-1]['code']})")
        print(f"   平均PE: {pe_df['pe_ratio'].mean():.2f}")
        print(f"   中位数PE: {pe_df['pe_ratio'].median():.2f}")
        
        # PE分布
        pe_ranges = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, float('inf'))]
        print(f"\n   PE分布:")
        for low, high in pe_ranges:
            count = len(pe_df[(pe_df['pe_ratio'] >= low) & (pe_df['pe_ratio'] < high)])
            if count > 0:
                print(f"     {low}-{high if high != float('inf') else '以上'}: {count}只股票")
    
    if not pb_df.empty:
        print(f"\n📊 市净率(PB)统计:")
        print(f"   最低PB: {pb_df['pb_ratio'].min():.2f} - {pb_df.iloc[0]['name']}({pb_df.iloc[0]['code']})")
        print(f"   最高PB: {pb_df['pb_ratio'].max():.2f} - {pb_df.iloc[-1]['name']}({pb_df.iloc[-1]['code']})")
        print(f"   平均PB: {pb_df['pb_ratio'].mean():.2f}")
        print(f"   中位数PB: {pb_df['pb_ratio'].median():.2f}")
        
        # PB分布
        pb_ranges = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, float('inf'))]
        print(f"\n   PB分布:")
        for low, high in pb_ranges:
            count = len(pb_df[(pb_df['pb_ratio'] >= low) & (pb_df['pb_ratio'] < high)])
            if count > 0:
                print(f"     {low}-{high if high != float('inf') else '以上'}: {count}只股票")

def save_to_csv(pe_df, pb_df, failed_codes):
    """
    将数据保存为CSV文件
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建输出目录
    output_dir = "stock_pe_pb_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存市盈率排序结果
    if not pe_df.empty:
        pe_filename = f"{output_dir}/A股市盈率排序_{timestamp}.csv"
        pe_df.to_csv(pe_filename, index=False, encoding='utf-8-sig')
        print(f"✅ 市盈率排序结果已保存: {pe_filename}")
    
    # 保存市净率排序结果
    if not pb_df.empty:
        pb_filename = f"{output_dir}/A股市净率排序_{timestamp}.csv"
        pb_df.to_csv(pb_filename, index=False, encoding='utf-8-sig')
        print(f"✅ 市净率排序结果已保存: {pb_filename}")
    
    # 保存失败列表
    if failed_codes:
        failed_filename = f"{output_dir}/数据获取失败列表_{timestamp}.csv"
        failed_df = pd.DataFrame(failed_codes, columns=['failed_code'])
        failed_df.to_csv(failed_filename, index=False, encoding='utf-8-sig')
        print(f"⚠️  失败列表已保存: {failed_filename}")
    
    # 保存综合文件（包含PE和PB排名）
    if not pe_df.empty and not pb_df.empty:
        # 合并PE和PB数据
        merged_df = pd.merge(
            pe_df[['code', 'name', 'pe_ratio', 'pe_rank', 'pe_percentile']],
            pb_df[['code', 'name', 'pb_ratio', 'pb_rank', 'pb_percentile']],
            on=['code', 'name'],
            how='outer'
        )
        
        # 计算综合排名（PE和PB排名的平均值）
        merged_df['combined_rank'] = (merged_df['pe_rank'].fillna(len(pe_df)) + 
                                    merged_df['pb_rank'].fillna(len(pb_df))) / 2
        merged_df = merged_df.sort_values('combined_rank').reset_index(drop=True)
        
        combined_filename = f"{output_dir}/A股PE_PB综合排序_{timestamp}.csv"
        merged_df.to_csv(combined_filename, index=False, encoding='utf-8-sig')
        print(f"✅ 综合排序结果已保存: {combined_filename}")

def main():
    """
    主函数
    """
    print("🚀 A股市盈率市净率数据抓取程序启动")
    print("="*60)
    
    # 获取股票列表
    stock_list = get_all_stock_codes()
    if stock_list is None or stock_list.empty:
        print("❌ 无法获取股票列表，程序退出")
        return
    
    stock_codes = stock_list['code'].tolist()
    
    # 选择获取数据的方法
    print("\n请选择数据获取方式:")
    print("1. 快速方法（推荐，速度快但可能数据不全）")
    print("2. 详细方法（速度慢但数据更完整）")
    
    choice = input("请输入选择 (1/2，默认1): ").strip() or "1"
    
    if choice == "1":
        # 使用快速方法
        df, failed_codes = get_stock_pe_pb_alternative()
    else:
        # 使用详细方法
        df, failed_codes = get_stock_pe_pb_batch(stock_codes)
    
    if df.empty:
        print("❌ 未能获取到有效数据，程序退出")
        return
    
    # 分析排序数据
    pe_sorted, pb_sorted = analyze_and_sort_data(df)
    
    # 生成报告
    generate_summary_report(pe_sorted, pb_sorted)
    
    # 显示前10名
    print("\n🏆 市盈率最低的前10只股票:")
    print(pe_sorted[['code', 'name', 'pe_ratio', 'pe_rank']].head(10).to_string(index=False))
    
    print("\n🏆 市净率最低的前10只股票:")
    print(pb_sorted[['code', 'name', 'pb_ratio', 'pb_rank']].head(10).to_string(index=False))
    
    # 保存到CSV
    save_to_csv(pe_sorted, pb_sorted, failed_codes)
    
    print(f"\n🎉 程序执行完成！数据已保存到 stock_pe_pb_data 目录")

if __name__ == "__main__":
    main()