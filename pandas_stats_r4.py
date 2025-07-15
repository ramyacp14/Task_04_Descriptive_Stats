import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import time

def sanitize_filename(name):
    """Remove or replace invalid characters for Windows filenames."""
    return ''.join(c if c.isalnum() or c in (' ', '_') else '_' for c in name).strip()

def should_visualize_column(col_name, col_data, col_type):
    """Determine if a column should be visualized based on its characteristics."""
    col_lower = col_name.lower()
    
    # Skip if column is empty or all NaN
    if len(col_data) == 0 or col_data.isna().all():
        return False
    
    # Skip ID columns and timestamps
    if any(skip_term in col_lower for skip_term in ['id', 'timestamp', 'created', 'updated', 'url', 'link']):
        return False
    
    # Skip very high cardinality text columns (likely unique identifiers or text content)
    if col_type == 'text' and col_data.nunique() > 20:
        return False
    
    # Skip columns with mostly missing data
    total_len = len(col_data)
    if total_len > 0 and col_data.isna().sum() / total_len > 0.8:
        return False
    
    # Skip binary columns with extreme imbalance (>95% one value)
    if col_data.nunique() == 2:
        value_counts = col_data.value_counts()
        if len(value_counts) > 0 and value_counts.iloc[0] / value_counts.sum() > 0.95:
            return False
    
    return True

def describe_dataset(df, output_folder):
    summary = {}
    visualizations_created = 0
    
    for col in df.columns:
        col_data = df[col].dropna()
        print(f"\nAnalyzing column: {col}")
        safe_col = sanitize_filename(col)

        if pd.api.types.is_numeric_dtype(col_data):
            desc = col_data.describe()
            
            # Check if this is actually continuous numeric data or boolean/categorical
            if col_data.dtype == 'bool' or col_data.nunique() <= 10:
                # Handle boolean or low-cardinality data as categorical
                vc = col_data.value_counts()
                summary[col] = {
                    'type': 'categorical',
                    'unique': int(col_data.nunique()),
                    'top_5': vc.head(5).to_dict()
                }
                
                # Only visualize if it makes sense
                if should_visualize_column(col, col_data, 'categorical') and not vc.empty:
                    plt.figure(figsize=(6,4))
                    vc.plot(kind='bar', color='lightgreen', edgecolor='black')
                    plt.title(f'Distribution of {col}')
                    plt.ylabel('Count')
                    plt.xticks(rotation=30, ha='right')
                    plt.tight_layout()
                    path = os.path.join(output_folder, f"{safe_col}_distribution.png")
                    plt.savefig(path)
                    plt.close()
                    print(f"  ✓ Saved bar chart to {path}")
                    visualizations_created += 1
                else:
                    print(f"  ⊘ Skipped visualization (not useful for this column)")
                    
            else:
                # Handle continuous numeric data
                summary[col] = {
                    'type': 'numeric',
                    'count': int(desc['count']),
                    'mean': desc['mean'],
                    'min': desc['min'],
                    'max': desc['max'],
                    'std_dev': desc['std'],
                }
                
                # Only visualize if it makes sense
                if should_visualize_column(col, col_data, 'numeric'):
                    plt.figure(figsize=(6,4))
                    col_data.hist(bins=20, color='skyblue', edgecolor='black')
                    plt.title(f'Histogram of {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    plt.tight_layout()
                    path = os.path.join(output_folder, f"{safe_col}_histogram.png")
                    plt.savefig(path)
                    plt.close()
                    print(f"  ✓ Saved histogram to {path}")
                    visualizations_created += 1
                else:
                    print(f"  ⊘ Skipped visualization (not useful for this column)")

        else:
            vc = col_data.value_counts()
            summary[col] = {
                'type': 'text',
                'unique': int(col_data.nunique()),
                'top_5': vc.head(5).to_dict()
            }
            
            # Only visualize if it makes sense
            if should_visualize_column(col, col_data, 'text') and not vc.empty:
                plt.figure(figsize=(8,4))  # Wider for text labels
                vc.head(10).plot(kind='bar', color='lightcoral', edgecolor='black')
                plt.title(f'Top 10 most common values in {col}')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                path = os.path.join(output_folder, f"{safe_col}_top10.png")
                plt.savefig(path)
                plt.close()
                print(f"  ✓ Saved bar chart to {path}")
                visualizations_created += 1
            else:
                print(f"  ⊘ Skipped visualization (not useful for this column)")
    
    print(f"\nTotal visualizations created: {visualizations_created}")
    return summary

def analyze_groups(df, group_cols):
    grouped = df.groupby(group_cols).size()
    sizes = grouped.values
    return {
        'group_by': group_cols,
        'total_groups': len(grouped),
        'avg_group_size': sizes.mean() if sizes.size else 0,
        'min_group_size': sizes.min() if sizes.size else 0,
        'max_group_size': sizes.max() if sizes.size else 0
    }

def analyze_file(filename):
    print(f"\n{'='*50}")
    print(f"Analyzing: {filename}")
    df = pd.read_csv(filename)
    output_folder = filename.replace('.csv', '_plots_pandas')
    os.makedirs(output_folder, exist_ok=True)

    t0 = time.time()
    overall_summary = describe_dataset(df, output_folder)
    t1 = time.time()

    print(f"\n{'Overall Dataset Summary':^50}")
    print("-"*50)
    for col, info in overall_summary.items():
        print(f"{col}:")
        if info['type']=='numeric':
            print(f"  Count={info['count']}, Mean={info['mean']:.2f}, Min={info['min']}, Max={info['max']}, StdDev={info['std_dev']:.2f}")
        else:
            print(f"  Unique={info['unique']}, Top5={info['top_5']}")

    print(f"\nTime taken: {t1 - t0:.2f} seconds")

    # Grouping (only if columns exist)
    possible_groupings = [['page_id'], ['page_id', 'ad_id']]
    for group_cols in possible_groupings:
        if all(col in df.columns for col in group_cols):
            stats = analyze_groups(df, group_cols)
            print(f"\n{'-'*50}")
            print(f"Grouped by {group_cols}:")
            print(f"  Total groups: {stats['total_groups']}")
            print(f"  Avg group size: {stats['avg_group_size']:.1f}")
            print(f"  Min group size: {stats['min_group_size']}")
            print(f"  Max group size: {stats['max_group_size']}")

    # Save summary to JSON
    output = {
        'file': filename,
        'num_rows': len(df),
        'summary': overall_summary,
        'time_taken': t1 - t0
    }
    outname = filename.replace('.csv', '_summary_pandas.json')
    with open(outname, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved summary to {outname}")

def main():
    files = [
        '2024_fb_ads_president_scored_anon.csv',
        '2024_fb_posts_president_scored_anon.csv',
        '2024_tw_posts_president_scored_anon.csv'
    ]
    for f in files:
        analyze_file(f)

if __name__ == "__main__":
    main()