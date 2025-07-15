import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import time
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class EnhancedPolarsAnalyzer:
    """Enhanced Polars-based data analyzer with comprehensive visualizations"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        self.analysis_results = {}
        self.performance_metrics = {}
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def load_dataset(self, file_path: Union[str, Path], **kwargs) -> pl.DataFrame:
        """Load dataset with enhanced error handling and performance tracking"""
        start_time = time.time()
        
        try:
            # Default Polars CSV reading parameters for better performance
            default_params = {
                'ignore_errors': True,
                'null_values': ['', 'NULL', 'null', 'NA', 'n/a', 'N/A'],
                'try_parse_dates': True,
                'infer_schema_length': 10000,
            }
            default_params.update(kwargs)
            
            df = pl.read_csv(file_path, **default_params)
            load_time = time.time() - start_time
            
            print(f"✓ Loaded {file_path}")
            print(f"  Shape: {df.height:,} rows × {df.width} columns")
            print(f"  Memory: {df.estimated_size():,} bytes")
            print(f"  Load time: {load_time:.3f}s")
            
            # Store performance metrics
            self.performance_metrics[str(file_path)] = {
                'load_time': load_time,
                'rows': df.height,
                'columns': df.width,
                'memory_bytes': df.estimated_size()
            }
            
            return df
            
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            return pl.DataFrame()
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return pl.DataFrame()

    def analyze_column_types(self, df: pl.DataFrame) -> Dict[str, List[str]]:
        """Categorize columns by data type for targeted analysis"""
        type_groups = {
            'numeric': [],
            'text': [],
            'datetime': [],
            'boolean': [],
            'list': [],
            'struct': []
        }
        
        for col in df.columns:
            dtype = df[col].dtype
            
            if dtype.is_numeric():
                type_groups['numeric'].append(col)
            elif dtype == pl.Boolean:
                type_groups['boolean'].append(col)
            elif dtype.is_temporal():
                type_groups['datetime'].append(col)
            elif dtype == pl.List:
                type_groups['list'].append(col)
            elif dtype == pl.Struct:
                type_groups['struct'].append(col)
            else:
                type_groups['text'].append(col)
        
        return type_groups

    def analyze_numeric_column(self, df: pl.DataFrame, col_name: str) -> Dict[str, Any]:
        """Comprehensive numeric column analysis"""
        try:
            # Use Polars expressions for efficient computation
            stats = df.select([
                pl.col(col_name).count().alias('count'),
                pl.col(col_name).null_count().alias('null_count'),
                pl.col(col_name).mean().alias('mean'),
                pl.col(col_name).median().alias('median'),
                pl.col(col_name).std().alias('std'),
                pl.col(col_name).var().alias('variance'),
                pl.col(col_name).min().alias('min'),
                pl.col(col_name).max().alias('max'),
                pl.col(col_name).quantile(0.25).alias('q1'),
                pl.col(col_name).quantile(0.75).alias('q3'),
                pl.col(col_name).sum().alias('sum'),
                pl.col(col_name).n_unique().alias('unique_values')
            ]).to_dicts()[0]
            
            # Calculate additional metrics
            stats['null_percentage'] = (stats['null_count'] / df.height) * 100
            stats['iqr'] = stats['q3'] - stats['q1']
            stats['range'] = stats['max'] - stats['min']
            
            # Detect outliers using IQR method
            if stats['iqr'] > 0:
                outlier_bounds = df.select([
                    (pl.col(col_name) < (stats['q1'] - 1.5 * stats['iqr'])).sum().alias('lower_outliers'),
                    (pl.col(col_name) > (stats['q3'] + 1.5 * stats['iqr'])).sum().alias('upper_outliers')
                ]).to_dicts()[0]
                
                stats['outliers'] = outlier_bounds['lower_outliers'] + outlier_bounds['upper_outliers']
                stats['outlier_percentage'] = (stats['outliers'] / stats['count']) * 100
            
            return stats
            
        except Exception as e:
            return {'error': f"Error analyzing numeric column {col_name}: {str(e)}"}

    def analyze_text_column(self, df: pl.DataFrame, col_name: str) -> Dict[str, Any]:
        """Comprehensive text column analysis"""
        try:
            # Basic stats
            basic_stats = df.select([
                pl.col(col_name).count().alias('count'),
                pl.col(col_name).null_count().alias('null_count'),
                pl.col(col_name).n_unique().alias('unique_values')
            ]).to_dicts()[0]
            
            basic_stats['null_percentage'] = (basic_stats['null_count'] / df.height) * 100
            basic_stats['uniqueness_ratio'] = basic_stats['unique_values'] / basic_stats['count']
            
            # Value counts for top values
            value_counts = df.select(col_name).filter(
                pl.col(col_name).is_not_null()
            ).to_series().value_counts().sort('count', descending=True)
            
            if value_counts.height > 0:
                top_values = {}
                for i in range(min(10, value_counts.height)):
                    val = value_counts[col_name][i]
                    count = value_counts['count'][i]
                    top_values[str(val)] = {
                        'count': count,
                        'percentage': (count / basic_stats['count']) * 100
                    }
                
                basic_stats['top_values'] = top_values
                basic_stats['most_common_value'] = str(value_counts[col_name][0])
                basic_stats['most_common_count'] = value_counts['count'][0]
            
            # String length analysis for text columns
            if basic_stats['count'] > 0:
                length_stats = df.select([
                    pl.col(col_name).str.len_chars().mean().alias('avg_length'),
                    pl.col(col_name).str.len_chars().min().alias('min_length'),
                    pl.col(col_name).str.len_chars().max().alias('max_length'),
                    pl.col(col_name).str.len_chars().median().alias('median_length')
                ]).to_dicts()[0]
                
                basic_stats.update(length_stats)
            
            # Try to detect structured data (JSON, lists, etc.)
            sample_values = df.select(col_name).filter(
                pl.col(col_name).is_not_null()
            ).limit(100).to_series().to_list()
            
            structured_count = 0
            for val in sample_values:
                try:
                    ast.literal_eval(str(val))
                    structured_count += 1
                except:
                    pass
            
            if structured_count > len(sample_values) * 0.1:
                basic_stats['likely_structured'] = True
                basic_stats['structured_percentage'] = (structured_count / len(sample_values)) * 100
            
            return basic_stats
            
        except Exception as e:
            return {'error': f"Error analyzing text column {col_name}: {str(e)}"}

    def create_numeric_visualizations(self, df: pl.DataFrame, col_name: str, 
                                    dataset_name: str, stats: Dict[str, Any]):
        """Create comprehensive visualizations for numeric columns"""
        try:
            # Get non-null values
            values = df.select(col_name).filter(pl.col(col_name).is_not_null()).to_series().to_numpy()
            
            if len(values) == 0:
                return
            
            # Create subplot layout
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Numeric Analysis: {col_name} ({dataset_name})', fontsize=16, fontweight='bold')
            
            # 1. Histogram with density
            axes[0, 0].hist(values, bins=50, alpha=0.7, edgecolor='black', density=True)
            axes[0, 0].axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["mean"]:.3f}')
            axes[0, 0].axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f'Median: {stats["median"]:.3f}')
            axes[0, 0].set_title('Distribution with Mean/Median')
            axes[0, 0].set_xlabel('Value')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Box plot with outliers
            box_plot = axes[0, 1].boxplot(values, patch_artist=True, labels=[col_name])
            box_plot['boxes'][0].set_facecolor('lightblue')
            axes[0, 1].set_title('Box Plot (Outlier Detection)')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add outlier statistics
            if 'outliers' in stats:
                axes[0, 1].text(0.02, 0.98, f'Outliers: {stats["outliers"]} ({stats["outlier_percentage"]:.1f}%)',
                               transform=axes[0, 1].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            # 3. Q-Q plot for normality check
            from scipy import stats as scipy_stats
            scipy_stats.probplot(values, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot (Normality Check)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Summary statistics table
            axes[1, 1].axis('off')
            summary_data = [
                ['Count', f"{stats['count']:,}"],
                ['Mean', f"{stats['mean']:.3f}"],
                ['Std Dev', f"{stats['std']:.3f}"],
                ['Min', f"{stats['min']:.3f}"],
                ['25%', f"{stats['q1']:.3f}"],
                ['50%', f"{stats['median']:.3f}"],
                ['75%', f"{stats['q3']:.3f}"],
                ['Max', f"{stats['max']:.3f}"],
                ['Null %', f"{stats['null_percentage']:.1f}%"],
                ['Unique', f"{stats['unique_values']:,}"]
            ]
            
            table = axes[1, 1].table(cellText=summary_data, colLabels=['Statistic', 'Value'],
                                   cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            axes[1, 1].set_title('Summary Statistics')
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = self.viz_dir / f"{dataset_name}_{col_name}_numeric_analysis.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved numeric visualization: {viz_path}")
            
        except Exception as e:
            print(f"  ❌ Error creating numeric visualization for {col_name}: {e}")

    def create_text_visualizations(self, df: pl.DataFrame, col_name: str, 
                                 dataset_name: str, stats: Dict[str, Any]):
        """Create comprehensive visualizations for text columns"""
        try:
            if 'top_values' not in stats or len(stats['top_values']) == 0:
                return
            
            # Create subplot layout
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Text Analysis: {col_name} ({dataset_name})', fontsize=16, fontweight='bold')
            
            # 1. Top values bar chart
            top_values = stats['top_values']
            values = list(top_values.keys())[:10]
            counts = [top_values[val]['count'] for val in values]
            
            bars = axes[0, 0].bar(range(len(values)), counts, alpha=0.7)
            axes[0, 0].set_title('Top 10 Most Frequent Values')
            axes[0, 0].set_xlabel('Values')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_xticks(range(len(values)))
            axes[0, 0].set_xticklabels([str(v)[:15] + '...' if len(str(v)) > 15 else str(v) 
                                      for v in values], rotation=45, ha='right')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, count) in enumerate(zip(bars, counts)):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{count:,}', ha='center', va='bottom')
            
            # 2. String length distribution (if available)
            if 'avg_length' in stats:
                length_values = df.select(col_name).filter(
                    pl.col(col_name).is_not_null()
                ).with_columns(
                    pl.col(col_name).str.len_chars().alias('length')
                ).select('length').to_series().to_numpy()
                
                axes[0, 1].hist(length_values, bins=30, alpha=0.7, edgecolor='black')
                axes[0, 1].axvline(stats['avg_length'], color='red', linestyle='--', 
                                 linewidth=2, label=f'Mean: {stats["avg_length"]:.1f}')
                axes[0, 1].set_title('String Length Distribution')
                axes[0, 1].set_xlabel('Character Length')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].axis('off')
                axes[0, 1].text(0.5, 0.5, 'Length analysis\nnot available', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
            
            # 3. Pie chart for top values
            pie_values = list(top_values.values())[:8]  # Top 8 values
            pie_labels = [str(k)[:20] + '...' if len(str(k)) > 20 else str(k) 
                         for k in list(top_values.keys())[:8]]
            pie_counts = [v['count'] for v in pie_values]
            
            axes[1, 0].pie(pie_counts, labels=pie_labels, autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('Distribution of Top Values')
            
            # 4. Summary statistics table
            axes[1, 1].axis('off')
            summary_data = [
                ['Total Count', f"{stats['count']:,}"],
                ['Unique Values', f"{stats['unique_values']:,}"],
                ['Uniqueness Ratio', f"{stats['uniqueness_ratio']:.1%}"],
                ['Null Percentage', f"{stats['null_percentage']:.1f}%"],
                ['Most Common', f"{stats['most_common_value'][:20]}..."],
                ['Most Common Count', f"{stats['most_common_count']:,}"]
            ]
            
            if 'avg_length' in stats:
                summary_data.extend([
                    ['Avg Length', f"{stats['avg_length']:.1f}"],
                    ['Min Length', f"{stats['min_length']}"],
                    ['Max Length', f"{stats['max_length']}"]
                ])
            
            table = axes[1, 1].table(cellText=summary_data, colLabels=['Statistic', 'Value'],
                                   cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            axes[1, 1].set_title('Summary Statistics')
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = self.viz_dir / f"{dataset_name}_{col_name}_text_analysis.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved text visualization: {viz_path}")
            
        except Exception as e:
            print(f"  ❌ Error creating text visualization for {col_name}: {e}")

    def create_dataset_overview_visualization(self, analysis: Dict[str, Any], dataset_name: str):
        """Create overview visualization for the entire dataset"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Dataset Overview: {dataset_name}', fontsize=16, fontweight='bold')
            
            # 1. Column types distribution
            col_types = analysis['overview']['column_types']
            type_names = [k for k, v in col_types.items() if v > 0]
            type_counts = [v for v in col_types.values() if v > 0]
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(type_names)))
            axes[0, 0].pie(type_counts, labels=type_names, autopct='%1.1f%%', colors=colors)
            axes[0, 0].set_title('Column Types Distribution')
            
            # 2. Data quality metrics
            quality = analysis['data_quality']
            quality_metrics = ['Complete', 'Null Values', 'Columns with Nulls', 'Duplicate Rows']
            quality_values = [
                quality['completeness_percentage'],
                (quality['total_null_values'] / (analysis['overview']['total_rows'] * 
                                               analysis['overview']['total_columns'])) * 100,
                (quality['columns_with_nulls'] / analysis['overview']['total_columns']) * 100,
                (quality['duplicate_rows'] / analysis['overview']['total_rows']) * 100
            ]
            
            bars = axes[0, 1].bar(quality_metrics, quality_values, color=['green', 'red', 'orange', 'yellow'])
            axes[0, 1].set_title('Data Quality Metrics (%)')
            axes[0, 1].set_ylabel('Percentage')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, quality_values):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.1f}%', ha='center', va='bottom')
            
            # 3. Memory usage and row distribution
            overview = analysis['overview']
            axes[1, 0].axis('off')
            overview_data = [
                ['Dataset', dataset_name],
                ['Total Rows', f"{overview['total_rows']:,}"],
                ['Total Columns', f"{overview['total_columns']:,}"],
                ['Memory Usage', f"{overview['memory_usage_bytes']:,} bytes"],
                ['Analysis Time', f"{analysis['analysis_duration']:.3f}s"]
            ]
            
            table = axes[1, 0].table(cellText=overview_data, colLabels=['Metric', 'Value'],
                                   cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            axes[1, 0].set_title('Dataset Overview')
            
            # 4. Null percentage by column (top 10 columns with most nulls)
            null_data = []
            for col, data in analysis['column_analyses'].items():
                if 'null_percentage' in data and data['null_percentage'] > 0:
                    null_data.append((col, data['null_percentage']))
            
            if null_data:
                null_data.sort(key=lambda x: x[1], reverse=True)
                top_null_cols = null_data[:10]
                
                col_names = [col[:15] + '...' if len(col) > 15 else col for col, _ in top_null_cols]
                null_percentages = [perc for _, perc in top_null_cols]
                
                axes[1, 1].barh(range(len(col_names)), null_percentages, color='red', alpha=0.7)
                axes[1, 1].set_title('Top 10 Columns with Null Values')
                axes[1, 1].set_xlabel('Null Percentage (%)')
                axes[1, 1].set_yticks(range(len(col_names)))
                axes[1, 1].set_yticklabels(col_names)
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].axis('off')
                axes[1, 1].text(0.5, 0.5, 'No null values\nfound in dataset', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = self.viz_dir / f"{dataset_name}_overview.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved overview visualization: {viz_path}")
            
        except Exception as e:
            print(f"  ❌ Error creating overview visualization: {e}")

    def create_correlation_heatmap(self, df: pl.DataFrame, dataset_name: str):
        """Create correlation heatmap for numeric columns"""
        try:
            # Get numeric columns
            numeric_cols = [col for col in df.columns if df[col].dtype.is_numeric()]
            
            if len(numeric_cols) < 2:
                print(f"  ⚠️  Not enough numeric columns for correlation analysis")
                return
            
            # Calculate correlation matrix using Polars
            numeric_df = df.select(numeric_cols)
            
            # Convert to pandas for correlation calculation (more efficient)
            import pandas as pd
            pandas_df = numeric_df.to_pandas()
            corr_matrix = pandas_df.corr()
            
            # Create heatmap
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": .5})
            
            plt.title(f'Correlation Matrix: {dataset_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save visualization
            viz_path = self.viz_dir / f"{dataset_name}_correlation_heatmap.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved correlation heatmap: {viz_path}")
            
        except Exception as e:
            print(f"  ❌ Error creating correlation heatmap: {e}")

    def create_group_analysis_visualization(self, df: pl.DataFrame, group_cols: List[str], 
                                          dataset_name: str, analysis_result: Dict[str, Any]):
        """Create visualizations for group analysis"""
        try:
            if 'error' in analysis_result:
                return
            
            # Group sizes visualization
            group_sizes = df.group_by(group_cols).len().sort('len', descending=True)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Group Analysis: {" + ".join(group_cols)} ({dataset_name})', 
                        fontsize=16, fontweight='bold')
            
            # 1. Distribution of group sizes
            sizes = group_sizes['len'].to_numpy()
            axes[0, 0].hist(sizes, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Distribution of Group Sizes')
            axes[0, 0].set_xlabel('Group Size')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Top 20 largest groups
            top_groups = group_sizes.head(20)
            group_labels = [f"Group {i+1}" for i in range(len(top_groups))]
            
            axes[0, 1].bar(range(len(top_groups)), top_groups['len'].to_numpy(), alpha=0.7)
            axes[0, 1].set_title('Top 20 Largest Groups')
            axes[0, 1].set_xlabel('Group Rank')
            axes[0, 1].set_ylabel('Group Size')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Cumulative distribution
            sorted_sizes = np.sort(sizes)
            cumulative = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
            axes[1, 0].plot(sorted_sizes, cumulative, linewidth=2)
            axes[1, 0].set_title('Cumulative Distribution of Group Sizes')
            axes[1, 0].set_xlabel('Group Size')
            axes[1, 0].set_ylabel('Cumulative Percentage')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Summary statistics
            axes[1, 1].axis('off')
            stats = analysis_result['group_size_analysis']['group_size_stats']
            summary_data = [
                ['Total Groups', f"{analysis_result['group_size_analysis']['total_groups']:,}"],
                ['Mean Size', f"{stats['mean']:.1f}"],
                ['Median Size', f"{stats['median']:.1f}"],
                ['Std Dev', f"{stats['std']:.1f}"],
                ['Min Size', f"{stats['min']:,}"],
                ['Max Size', f"{stats['max']:,}"],
                ['Total Records', f"{analysis_result['group_size_analysis']['total_records']:,}"]
            ]
            
            table = axes[1, 1].table(cellText=summary_data, colLabels=['Statistic', 'Value'],
                                   cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            axes[1, 1].set_title('Group Size Statistics')
            
            plt.tight_layout()
            
            # Save visualization
            group_name = "_".join(group_cols)
            viz_path = self.viz_dir / f"{dataset_name}_{group_name}_group_analysis.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved group analysis visualization: {viz_path}")
            
        except Exception as e:
            print(f"  ❌ Error creating group analysis visualization: {e}")

    def perform_comprehensive_analysis(self, df: pl.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Main analysis function with comprehensive statistics and visualizations"""
        if df.is_empty():
            return {'error': 'Dataset is empty'}
        
        start_time = time.time()
        
        print(f"\n🎨 Creating visualizations for {dataset_name}...")
        
        # Dataset overview
        overview = {
            'dataset_name': dataset_name,
            'total_rows': df.height,
            'total_columns': df.width,
            'memory_usage_bytes': df.estimated_size(),
            'column_names': df.columns
        }
        
        # Categorize columns
        column_types = self.analyze_column_types(df)
        overview['column_types'] = {k: len(v) for k, v in column_types.items()}
        
        # Analyze each column
        column_analyses = {}
        
        # Numeric columns
        for col in column_types['numeric']:
            print(f"  📊 Analyzing numeric column: {col}")
            column_analyses[col] = self.analyze_numeric_column(df, col)
            column_analyses[col]['type'] = 'numeric'
            
            # Create visualizations for numeric columns
            if 'error' not in column_analyses[col]:
                self.create_numeric_visualizations(df, col, dataset_name, column_analyses[col])
        
        # Text columns
        for col in column_types['text']:
            print(f"  📝 Analyzing text column: {col}")
            column_analyses[col] = self.analyze_text_column(df, col)
            column_analyses[col]['type'] = 'text'
            
            # Create visualizations for text columns
            if 'error' not in column_analyses[col]:
                self.create_text_visualizations(df, col, dataset_name, column_analyses[col])
        
        # Boolean columns
        for col in column_types['boolean']:
            print(f"  ✅ Analyzing boolean column: {col}")
            bool_stats = df.select([
                pl.col(col).count().alias('count'),
                pl.col(col).null_count().alias('null_count'),
                pl.col(col).sum().alias('true_count')
            ]).to_dicts()[0]
            
            bool_stats['false_count'] = bool_stats['count'] - bool_stats['true_count']
            bool_stats['true_percentage'] = (bool_stats['true_count'] / bool_stats['count']) * 100
            bool_stats['type'] = 'boolean'
            column_analyses[col] = bool_stats
            
            # Create boolean visualization
            self.create_boolean_visualization(df, col, dataset_name, bool_stats)
        
        # DateTime columns
        for col in column_types['datetime']:
            print(f"  📅 Analyzing datetime column: {col}")
            datetime_stats = self.analyze_datetime_column(df, col)
            datetime_stats['type'] = 'datetime'
            column_analyses[col] = datetime_stats
            
            # Create datetime visualization
            if 'error' not in datetime_stats:
                self.create_datetime_visualization(df, col, dataset_name, datetime_stats)
        
        # Overall data quality assessment
        total_cells = df.height * df.width
        total_nulls = sum(df.select(pl.col(col).null_count()).item() for col in df.columns)
        
        data_quality = {
            'completeness_percentage': ((total_cells - total_nulls) / total_cells) * 100,
            'total_null_values': total_nulls,
            'columns_with_nulls': sum(1 for col in df.columns if df.select(pl.col(col).null_count()).item() > 0),
            'duplicate_rows': df.height - df.unique().height
        }
        
        analysis_time = time.time() - start_time
        
        analysis_result = {
            'overview': overview,
            'column_analyses': column_analyses,
            'data_quality': data_quality,
            'column_type_distribution': column_types,
            'analysis_duration': analysis_time
        }
        
        # Create overview visualization
        self.create_dataset_overview_visualization(analysis_result, dataset_name)
        
        # Create correlation heatmap
        self.create_correlation_heatmap(df, dataset_name)
        
        return analysis_result

    def analyze_datetime_column(self, df: pl.DataFrame, col_name: str) -> Dict[str, Any]:
        """Comprehensive datetime column analysis"""
        try:
            # Basic datetime stats
            stats = df.select([
                pl.col(col_name).count().alias('count'),
                pl.col(col_name).null_count().alias('null_count'),
                pl.col(col_name).min().alias('min_date'),
                pl.col(col_name).max().alias('max_date'),
                pl.col(col_name).n_unique().alias('unique_values')
            ]).to_dicts()[0]
            
            stats['null_percentage'] = (stats['null_count'] / df.height) * 100
            stats['date_range_days'] = (stats['max_date'] - stats['min_date']).days if stats['max_date'] and stats['min_date'] else 0
            
            return stats
            
        except Exception as e:
            return {'error': f"Error analyzing datetime column {col_name}: {str(e)}"}

    def create_boolean_visualization(self, df: pl.DataFrame, col_name: str, 
                                   dataset_name: str, stats: Dict[str, Any]):
        """Create visualization for boolean columns"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'Boolean Analysis: {col_name} ({dataset_name})', fontsize=16, fontweight='bold')
            
            # 1. Pie chart
            labels = ['True', 'False']
            sizes = [stats['true_count'], stats['false_count']]
            colors = ['lightgreen', 'lightcoral']
            
            axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            axes[0].set_title('True/False Distribution')
            
            # 2. Bar chart with counts
            axes[1].bar(labels, sizes, color=colors, alpha=0.7)
            axes[1].set_title('True/False Counts')
            axes[1].set_ylabel('Count')
            
            # Add value labels on bars
            for i, (label, count) in enumerate(zip(labels, sizes)):
                axes[1].text(i, count, f'{count:,}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = self.viz_dir / f"{dataset_name}_{col_name}_boolean_analysis.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved boolean visualization: {viz_path}")
            
        except Exception as e:
            print(f"  ❌ Error creating boolean visualization for {col_name}: {e}")

    def create_datetime_visualization(self, df: pl.DataFrame, col_name: str, 
                                    dataset_name: str, stats: Dict[str, Any]):
        """Create visualization for datetime columns"""
        try:
            # Get datetime values
            datetime_df = df.select([
                pl.col(col_name).alias('datetime'),
                pl.col(col_name).dt.year().alias('year'),
                pl.col(col_name).dt.month().alias('month'),
                pl.col(col_name).dt.day().alias('day'),
                pl.col(col_name).dt.weekday().alias('weekday')
            ]).filter(pl.col('datetime').is_not_null())
            
            if datetime_df.height == 0:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'DateTime Analysis: {col_name} ({dataset_name})', fontsize=16, fontweight='bold')
            
            # 1. Timeline plot
            dates = datetime_df.select('datetime').to_series().to_numpy()
            axes[0, 0].hist(dates, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Timeline Distribution')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Year distribution
            year_counts = datetime_df.group_by('year').len().sort('year')
            if year_counts.height > 0:
                axes[0, 1].bar(year_counts['year'].to_numpy(), year_counts['len'].to_numpy(), alpha=0.7)
                axes[0, 1].set_title('Distribution by Year')
                axes[0, 1].set_xlabel('Year')
                axes[0, 1].set_ylabel('Count')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. Month distribution
            month_counts = datetime_df.group_by('month').len().sort('month')
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            if month_counts.height > 0:
                months = month_counts['month'].to_numpy()
                counts = month_counts['len'].to_numpy()
                axes[1, 0].bar(months, counts, alpha=0.7)
                axes[1, 0].set_title('Distribution by Month')
                axes[1, 0].set_xlabel('Month')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].set_xticks(range(1, 13))
                axes[1, 0].set_xticklabels(month_names, rotation=45)
            
            # 4. Weekday distribution
            weekday_counts = datetime_df.group_by('weekday').len().sort('weekday')
            weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            if weekday_counts.height > 0:
                weekdays = weekday_counts['weekday'].to_numpy()
                counts = weekday_counts['len'].to_numpy()
                axes[1, 1].bar(weekdays, counts, alpha=0.7)
                axes[1, 1].set_title('Distribution by Weekday')
                axes[1, 1].set_xlabel('Weekday')
                axes[1, 1].set_ylabel('Count')
                axes[1, 1].set_xticks(range(1, 8))
                axes[1, 1].set_xticklabels(weekday_names)
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = self.viz_dir / f"{dataset_name}_{col_name}_datetime_analysis.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved datetime visualization: {viz_path}")
            
        except Exception as e:
            print(f"  ❌ Error creating datetime visualization for {col_name}: {e}")

    def create_performance_dashboard(self):
        """Create a performance dashboard showing analysis metrics"""
        try:
            if not self.performance_metrics:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Performance Dashboard', fontsize=16, fontweight='bold')
            
            datasets = list(self.performance_metrics.keys())
            load_times = [self.performance_metrics[d]['load_time'] for d in datasets]
            row_counts = [self.performance_metrics[d]['rows'] for d in datasets]
            memory_usage = [self.performance_metrics[d]['memory_bytes'] for d in datasets]
            
            # 1. Load times
            axes[0, 0].bar(range(len(datasets)), load_times, alpha=0.7)
            axes[0, 0].set_title('Dataset Load Times')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].set_xticks(range(len(datasets)))
            axes[0, 0].set_xticklabels([d.split('/')[-1][:15] for d in datasets], rotation=45)
            
            # 2. Row counts
            axes[0, 1].bar(range(len(datasets)), row_counts, alpha=0.7, color='green')
            axes[0, 1].set_title('Dataset Row Counts')
            axes[0, 1].set_ylabel('Number of Rows')
            axes[0, 1].set_xticks(range(len(datasets)))
            axes[0, 1].set_xticklabels([d.split('/')[-1][:15] for d in datasets], rotation=45)
            
            # 3. Memory usage
            axes[1, 0].bar(range(len(datasets)), memory_usage, alpha=0.7, color='red')
            axes[1, 0].set_title('Memory Usage')
            axes[1, 0].set_ylabel('Bytes')
            axes[1, 0].set_xticks(range(len(datasets)))
            axes[1, 0].set_xticklabels([d.split('/')[-1][:15] for d in datasets], rotation=45)
            
            # 4. Processing speed (rows per second)
            speeds = [row_counts[i] / load_times[i] for i in range(len(datasets))]
            axes[1, 1].bar(range(len(datasets)), speeds, alpha=0.7, color='orange')
            axes[1, 1].set_title('Processing Speed')
            axes[1, 1].set_ylabel('Rows per Second')
            axes[1, 1].set_xticks(range(len(datasets)))
            axes[1, 1].set_xticklabels([d.split('/')[-1][:15] for d in datasets], rotation=45)
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = self.viz_dir / "performance_dashboard.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved performance dashboard: {viz_path}")
            
        except Exception as e:
            print(f"  ❌ Error creating performance dashboard: {e}")

    def perform_group_analysis(self, df: pl.DataFrame, group_cols: List[str], 
                             agg_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Enhanced groupby analysis with multiple aggregation options"""
        if df.is_empty() or not group_cols:
            return {'error': 'Invalid input for groupby analysis'}
        
        # Validate columns exist
        missing_cols = [col for col in group_cols if col not in df.columns]
        if missing_cols:
            return {'error': f'Missing columns: {missing_cols}'}
        
        try:
            # Group sizes
            group_sizes = df.group_by(group_cols).len().sort('len', descending=True)
            
            size_stats = {
                'total_groups': group_sizes.height,
                'group_size_stats': {
                    'min': group_sizes['len'].min(),
                    'max': group_sizes['len'].max(),
                    'mean': group_sizes['len'].mean(),
                    'median': group_sizes['len'].median(),
                    'std': group_sizes['len'].std()
                },
                'total_records': group_sizes['len'].sum()
            }
            
            # Numeric aggregations
            numeric_cols = [col for col in df.columns 
                          if df[col].dtype.is_numeric() and col not in group_cols]
            
            if agg_cols:
                numeric_cols = [col for col in numeric_cols if col in agg_cols]
            
            numeric_aggregations = {}
            if numeric_cols:
                for col in numeric_cols:
                    try:
                        # Multiple aggregations at once
                        agg_result = df.group_by(group_cols).agg([
                            pl.col(col).count().alias('count'),
                            pl.col(col).mean().alias('mean'),
                            pl.col(col).std().alias('std'),
                            pl.col(col).min().alias('min'),
                            pl.col(col).max().alias('max'),
                            pl.col(col).median().alias('median'),
                            pl.col(col).sum().alias('sum')
                        ])
                        
                        # Summary statistics across groups
                        numeric_aggregations[col] = {
                            'groups_with_data': agg_result['count'].filter(pl.col('count') > 0).len(),
                            'mean_across_groups': agg_result['mean'].mean(),
                            'std_of_means': agg_result['mean'].std(),
                            'min_group_mean': agg_result['mean'].min(),
                            'max_group_mean': agg_result['mean'].max(),
                            'total_sum': agg_result['sum'].sum()
                        }
                    except Exception as e:
                        numeric_aggregations[col] = {'error': str(e)}
            
            return {
                'group_size_analysis': size_stats,
                'numeric_aggregations': numeric_aggregations,
                'grouping_columns': group_cols
            }
            
        except Exception as e:
            return {'error': f'Groupby analysis failed: {str(e)}'}

    def generate_summary_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a readable summary report"""
        if 'error' in analysis:
            return f"Analysis failed: {analysis['error']}"
        
        report = []
        report.append("=" * 80)
        report.append(f"DATASET ANALYSIS REPORT: {analysis['overview']['dataset_name']}")
        report.append("=" * 80)
        
        # Overview section
        overview = analysis['overview']
        report.append(f"\n📊 DATASET OVERVIEW")
        report.append(f"   Rows: {overview['total_rows']:,}")
        report.append(f"   Columns: {overview['total_columns']}")
        report.append(f"   Memory Usage: {overview['memory_usage_bytes']:,} bytes")
        report.append(f"   Analysis Time: {analysis['analysis_duration']:.3f} seconds")
        
        # Data quality section
        quality = analysis['data_quality']
        report.append(f"\n🔍 DATA QUALITY")
        report.append(f"   Completeness: {quality['completeness_percentage']:.1f}%")
        report.append(f"   Null Values: {quality['total_null_values']:,}")
        report.append(f"   Columns with Nulls: {quality['columns_with_nulls']}")
        report.append(f"   Duplicate Rows: {quality['duplicate_rows']:,}")
        
        # Column type distribution
        types = analysis['column_type_distribution']
        report.append(f"\n📋 COLUMN TYPES")
        for col_type, cols in types.items():
            if cols:
                report.append(f"   {col_type.title()}: {len(cols)} columns")
        
        # Visualization summary
        report.append(f"\n🎨 VISUALIZATIONS CREATED")
        report.append(f"   📈 Overview dashboard saved")
        report.append(f"   🔗 Correlation heatmap saved")
        
        numeric_count = len([col for col, data in analysis['column_analyses'].items() 
                           if data.get('type') == 'numeric'])
        text_count = len([col for col, data in analysis['column_analyses'].items() 
                        if data.get('type') == 'text'])
        
        report.append(f"   📊 {numeric_count} numeric column visualizations")
        report.append(f"   📝 {text_count} text column visualizations")
        
        # Sample numeric columns
        numeric_cols = [(col, data) for col, data in analysis['column_analyses'].items() 
                       if data.get('type') == 'numeric' and 'error' not in data]
        
        if numeric_cols:
            report.append(f"\n🔢 NUMERIC COLUMNS SAMPLE")
            for col, data in numeric_cols[:5]:  # Show first 5
                report.append(f"   {col}:")
                report.append(f"     Mean: {data['mean']:.3f}, Std: {data['std']:.3f}")
                report.append(f"     Range: {data['min']:.3f} to {data['max']:.3f}")
                report.append(f"     Nulls: {data['null_percentage']:.1f}%")
                if 'outliers' in data:
                    report.append(f"     Outliers: {data['outliers']} ({data['outlier_percentage']:.1f}%)")
        
        # Sample text columns
        text_cols = [(col, data) for col, data in analysis['column_analyses'].items() 
                    if data.get('type') == 'text' and 'error' not in data]
        
        if text_cols:
            report.append(f"\n📝 TEXT COLUMNS SAMPLE")
            for col, data in text_cols[:5]:  # Show first 5
                report.append(f"   {col}:")
                report.append(f"     Unique Values: {data['unique_values']:,}")
                report.append(f"     Uniqueness: {data['uniqueness_ratio']:.1%}")
                if 'most_common_value' in data:
                    report.append(f"     Most Common: '{data['most_common_value']}' ({data['most_common_count']} times)")
                if 'avg_length' in data:
                    report.append(f"     Avg Length: {data['avg_length']:.1f} chars")
        
        return "\n".join(report)

    def save_results(self, results: Dict[str, Any], filename: str):
        """Save analysis results to JSON file"""
        output_path = self.output_dir / filename
        
        # Make results JSON serializable
        json_safe_results = self._make_json_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(json_safe_results, f, indent=2, default=str)
        
        print(f"✓ Results saved to {output_path}")

    def _make_json_serializable(self, obj):
        """Convert complex objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj

def main():
    """Main execution function"""
    analyzer = EnhancedPolarsAnalyzer()
    
    # Configuration for datasets
    datasets = {
        'facebook_ads': '2024_fb_ads_president_scored_anon.csv',
        'facebook_posts': '2024_fb_posts_president_scored_anon.csv', 
        'twitter_posts': '2024_tw_posts_president_scored_anon.csv'
    }
    
    all_results = {}
    
    for dataset_name, file_path in datasets.items():
        print(f"\n{'='*80}")
        print(f"ANALYZING: {dataset_name}")
        print(f"{'='*80}")
        
        # Load dataset
        df = analyzer.load_dataset(file_path)
        
        if not df.is_empty():
            # Perform comprehensive analysis
            analysis = analyzer.perform_comprehensive_analysis(df, dataset_name)
            
            if 'error' not in analysis:
                all_results[dataset_name] = analysis
                
                # Display summary report
                print(analyzer.generate_summary_report(analysis))
                
                # Perform groupby analyses based on dataset
                if dataset_name == 'facebook_ads' and 'page_id' in df.columns:
                    print(f"\n🔍 GROUP ANALYSIS: Facebook Ads by Page")
                    print("-" * 60)
                    
                    page_analysis = analyzer.perform_group_analysis(df, ['page_id'])
                    if 'error' not in page_analysis:
                        stats = page_analysis['group_size_analysis']['group_size_stats']
                        print(f"   Unique pages: {page_analysis['group_size_analysis']['total_groups']:,}")
                        print(f"   Ads per page: {stats['mean']:.1f} ± {stats['std']:.1f}")
                        print(f"   Range: {stats['min']} to {stats['max']} ads")
                        
                        # Create group analysis visualization
                        analyzer.create_group_analysis_visualization(df, ['page_id'], dataset_name, page_analysis)
                    
                    # Multi-level grouping
                    if 'ad_id' in df.columns:
                        multi_analysis = analyzer.perform_group_analysis(df, ['page_id', 'ad_id'])
                        if 'error' not in multi_analysis:
                            print(f"   Page-Ad combinations: {multi_analysis['group_size_analysis']['total_groups']:,}")
                            analyzer.create_group_analysis_visualization(df, ['page_id', 'ad_id'], dataset_name, multi_analysis)
                
                elif dataset_name == 'twitter_posts' and 'user_id' in df.columns:
                    print(f"\n🔍 GROUP ANALYSIS: Twitter Posts by User")
                    print("-" * 60)
                    
                    user_analysis = analyzer.perform_group_analysis(df, ['user_id'])
                    if 'error' not in user_analysis:
                        stats = user_analysis['group_size_analysis']['group_size_stats']
                        print(f"   Unique users: {user_analysis['group_size_analysis']['total_groups']:,}")
                        print(f"   Posts per user: {stats['mean']:.1f} ± {stats['std']:.1f}")
                        print(f"   Range: {stats['min']} to {stats['max']} posts")
                        
                        # Create group analysis visualization
                        analyzer.create_group_analysis_visualization(df, ['user_id'], dataset_name, user_analysis)
                
                # Save individual results
                analyzer.save_results(analysis, f"{dataset_name}_analysis.json")
            
            else:
                print(f"❌ Analysis failed for {dataset_name}: {analysis['error']}")
    
    # Create performance dashboard
    analyzer.create_performance_dashboard()
    
    # Final summary
    if all_results:
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        
        total_rows = sum(r['overview']['total_rows'] for r in all_results.values())
        total_time = sum(r['analysis_duration'] for r in all_results.values())
        
        print(f"✓ Successfully analyzed {len(all_results)} datasets")
        print(f"✓ Total rows processed: {total_rows:,}")
        print(f"✓ Total analysis time: {total_time:.3f} seconds")
        print(f"✓ Processing speed: {total_rows/total_time:,.0f} rows/second")
        
        # Count visualizations created
        viz_count = len(list(analyzer.viz_dir.glob('*.png')))
        print(f"✓ Created {viz_count} visualizations in {analyzer.viz_dir}")
        
        # Save combined results
        analyzer.save_results(all_results, "complete_analysis_results.json")
        
        # Performance summary
        print(f"\n📈 PERFORMANCE METRICS")
        for dataset, metrics in analyzer.performance_metrics.items():
            print(f"   {dataset}: {metrics['rows']:,} rows in {metrics['load_time']:.3f}s")
        
        print(f"\n🎨 VISUALIZATION SUMMARY")
        print(f"   All visualizations saved to: {analyzer.viz_dir}")
        print(f"   📊 Dataset overviews: {len(all_results)}")
        print(f"   🔗 Correlation heatmaps: {len(all_results)}")
        print(f"   📈 Column-specific charts: {viz_count - len(all_results) * 2}")
        
    else:
        print("❌ No datasets were successfully analyzed")

if __name__ == "__main__":
    main()