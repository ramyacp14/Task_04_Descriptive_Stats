import csv
import math
import json
import re
from collections import Counter, defaultdict
import time
import matplotlib.pyplot as plt
import os

def sanitize_filename(name):
    """Remove or replace invalid characters for Windows filenames."""
    # Replace invalid characters with underscores
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', name)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    # Limit length to avoid path issues
    return sanitized[:100] if len(sanitized) > 100 else sanitized

def read_csv(filename):
    """Read CSV file with better error handling."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return []
    except UnicodeDecodeError:
        # Try different encodings
        try:
            with open(filename, 'r', encoding='utf-8-sig') as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            try:
                with open(filename, 'r', encoding='latin-1') as f:
                    return list(csv.DictReader(f))
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                return []

def is_number(s):
    """Check if string can be converted to float."""
    if not s or str(s).strip() == '':
        return False
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

def to_float(s):
    """Convert string to float with error handling."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0

def mean(numbers):
    """Calculate mean of a list of numbers."""
    return sum(numbers) / len(numbers) if numbers else 0.0

def std_dev(numbers):
    """Calculate standard deviation of a list of numbers."""
    if len(numbers) < 2:
        return 0.0
    avg = mean(numbers)
    variance = sum((x - avg) ** 2 for x in numbers) / (len(numbers) - 1)
    return math.sqrt(variance)

def should_visualize_column(col_name, real_values, col_type):
    """Determine if a column should be visualized based on its characteristics."""
    col_lower = col_name.lower()
    
    # Skip if column is empty
    if not real_values:
        return False
    
    # Skip ID columns, timestamps, and URLs
    skip_terms = ['id', 'timestamp', 'created', 'updated', 'url', 'link', 'href']
    if any(skip_term in col_lower for skip_term in skip_terms):
        return False
    
    # Skip very high cardinality text columns (likely unique identifiers or text content)
    if col_type == 'text':
        unique_count = len(set(real_values))
        if unique_count > 20:
            return False
    
    # Skip columns with extreme imbalance for binary data
    if col_type == 'text' and len(set(real_values)) == 2:
        counter = Counter(real_values)
        most_common_count = counter.most_common(1)[0][1]
        if most_common_count / len(real_values) > 0.95:
            return False
    
    return True

def describe_column(col_values, col_name, output_folder):
    """Analyze a single column and optionally create visualization."""
    # Filter out empty/null values
    real_values = [v for v in col_values if v and str(v).strip() != '']
    numeric_values = [to_float(v) for v in real_values if is_number(v)]
    
    result = {
        'total': len(col_values), 
        'missing': len(col_values) - len(real_values)
    }
    
    # Determine if column is mostly numeric (threshold: >50% numeric)
    if numeric_values and len(numeric_values) / len(real_values) > 0.5:
        # Check if it's actually categorical (boolean or very low cardinality)
        unique_count = len(set(numeric_values))
        if unique_count <= 10:
            # Treat as categorical
            counter = Counter(real_values)
            result.update({
                'type': 'categorical',
                'unique': len(counter),
                'top_5': counter.most_common(5)
            })
            
            # Visualization for categorical data
            if should_visualize_column(col_name, real_values, 'categorical'):
                safe_col_name = sanitize_filename(col_name)
                labels, counts = zip(*counter.most_common(10)) if counter else ([], [])
                
                if labels and counts:
                    plt.figure(figsize=(6, 4))
                    plt.bar(labels, counts, color='lightgreen', edgecolor='black')
                    plt.title(f'Distribution of {col_name}')
                    plt.ylabel('Count')
                    plt.xticks(rotation=30, ha='right')
                    plt.tight_layout()
                    plot_path = os.path.join(output_folder, f'{safe_col_name}_distribution.png')
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"  ✓ Saved bar chart to {plot_path}")
                else:
                    print(f"  ⊘ Skipped visualization (empty data)")
            else:
                print(f"  ⊘ Skipped visualization (not useful for this column)")
        else:
            # Treat as continuous numeric
            result.update({
                'type': 'numeric',
                'count': len(numeric_values),
                'mean': mean(numeric_values),
                'min': min(numeric_values),
                'max': max(numeric_values),
                'std_dev': std_dev(numeric_values)
            })
            
            # Visualization for numeric data
            if should_visualize_column(col_name, real_values, 'numeric'):
                safe_col_name = sanitize_filename(col_name)
                plt.figure(figsize=(6, 4))
                plt.hist(numeric_values, bins=20, color='skyblue', edgecolor='black')
                plt.title(f'Histogram of {col_name}')
                plt.xlabel(col_name)
                plt.ylabel('Frequency')
                plt.tight_layout()
                plot_path = os.path.join(output_folder, f'{safe_col_name}_histogram.png')
                plt.savefig(plot_path)
                plt.close()
                print(f"  ✓ Saved histogram to {plot_path}")
            else:
                print(f"  ⊘ Skipped visualization (not useful for this column)")
    else:
        # Mostly text/categorical
        counter = Counter(real_values)
        result.update({
            'type': 'text',
            'unique': len(counter),
            'top_5': counter.most_common(5)
        })
        
        # Visualization for text data
        if should_visualize_column(col_name, real_values, 'text'):
            safe_col_name = sanitize_filename(col_name)
            labels, counts = zip(*counter.most_common(10)) if counter else ([], [])
            
            if labels and counts:
                plt.figure(figsize=(8, 4))  # Wider for text labels
                plt.bar(labels, counts, color='lightcoral', edgecolor='black')
                plt.title(f'Top 10 most common values in {col_name}')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plot_path = os.path.join(output_folder, f'{safe_col_name}_top10.png')
                plt.savefig(plot_path)
                plt.close()
                print(f"  ✓ Saved bar chart to {plot_path}")
            else:
                print(f"  ⊘ Skipped visualization (empty data)")
        else:
            print(f"  ⊘ Skipped visualization (not useful for this column)")
    
    return result

def describe_dataset(rows, output_folder):
    """Analyze entire dataset and create summary."""
    if not rows:
        return {}
    
    summary = {}
    visualizations_created = 0
    
    for col in rows[0].keys():
        col_values = [row.get(col, '') for row in rows]
        print(f"\nAnalyzing column: {col}")
        summary[col] = describe_column(col_values, col, output_folder)
        
        # Count visualizations (rough estimate)
        if "✓" in str(summary[col]):
            visualizations_created += 1
    
    print(f"\nDataset analysis complete!")
    return summary

def group_by(rows, cols):
    """Group rows by specified columns."""
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row.get(col, '') for col in cols)
        groups[key].append(row)
    return groups

def analyze_groups(groups):
    """Analyze group statistics."""
    sizes = [len(g) for g in groups.values()]
    return {
        'total_groups': len(groups),
        'avg_group_size': mean(sizes),
        'min_group_size': min(sizes) if sizes else 0,
        'max_group_size': max(sizes) if sizes else 0
    }

def analyze_file(filename, group_cols_list=None):
    """Analyze a single CSV file."""
    if group_cols_list is None:
        group_cols_list = [['page_id'], ['page_id', 'ad_id']]
    
    print(f"\n{'='*50}")
    print(f"Analyzing: {filename}")
    data = read_csv(filename)
    
    if not data:
        print("No data found or unable to read file.")
        return
    
    # Create output folder
    output_folder = filename.replace('.csv', '_plots_python')
    os.makedirs(output_folder, exist_ok=True)
    
    # Analyze dataset
    t0 = time.time()
    overall_summary = describe_dataset(data, output_folder)
    t1 = time.time()
    
    # Print summary
    print(f"\n{'Overall Dataset Summary':^50}")
    print("-" * 50)
    for col, info in overall_summary.items():
        print(f"{col}:")
        if info['type'] == 'numeric':
            print(f"  Count={info['count']}, Mean={info['mean']:.2f}, "
                  f"Min={info['min']}, Max={info['max']}, StdDev={info['std_dev']:.2f}")
        elif info['type'] == 'categorical':
            print(f"  Unique={info['unique']}, Top5={info['top_5']}")
        else:  # text
            print(f"  Unique={info['unique']}, Top5={info['top_5']}")
    
    print(f"\nTime taken: {t1 - t0:.2f} seconds")
    
    # Group analysis
    for group_cols in group_cols_list:
        # Check if all grouping columns exist
        if all(col in data[0] for col in group_cols):
            groups = group_by(data, group_cols)
            group_stats = analyze_groups(groups)
            print(f"\n{'-'*50}")
            print(f"Grouped by {group_cols}:")
            print(f"  Total groups: {group_stats['total_groups']}")
            print(f"  Avg group size: {group_stats['avg_group_size']:.1f}")
            print(f"  Min group size: {group_stats['min_group_size']}")
            print(f"  Max group size: {group_stats['max_group_size']}")
    
    # Save summary to JSON
    output = {
        'file': filename,
        'num_rows': len(data),
        'summary': overall_summary,
        'time_taken': t1 - t0
    }
    
    outname = filename.replace('.csv', '_summary_python.json')
    try:
        with open(outname, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nSaved summary to {outname}")
    except Exception as e:
        print(f"Error saving summary: {e}")

def main():
    """Main function to analyze multiple files."""
    files = [
        '2024_fb_ads_president_scored_anon.csv',
        '2024_fb_posts_president_scored_anon.csv',
        '2024_tw_posts_president_scored_anon.csv'
    ]
    
    print("Starting CSV analysis with pure Python...")
    print("=" * 60)
    
    for f in files:
        if os.path.exists(f):
            analyze_file(f)
        else:
            print(f"\nFile not found: {f}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")

if __name__ == "__main__":
    main()