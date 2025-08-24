#!/usr/bin/env python3
"""
Generate comprehensive analysis report addressing assignment requirements:
- Comparison table
- Analysis of strengths and trade-offs
- Performance benchmarking
- Recommendations
Usage:
  python scripts/generate_report.py --eval reports/comprehensive.csv --out reports/analysis_report.md
"""
import argparse, pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_evaluation_data(eval_file):
    """Load evaluation data from CSV"""
    df = pd.read_csv(eval_file)
    print(f"Loaded {len(df)} evaluation records from {eval_file}")
    return df

def generate_comparison_table(df):
    """Generate comprehensive comparison table"""
    
    # Group by query type
    query_type_stats = {}
    for query_type in df['query_type'].unique():
        type_df = df[df['query_type'] == query_type]
        
        query_type_stats[query_type] = {
            'count': len(type_df),
            'rag_success_rate': type_df['rag_success'].mean() * 100,
            'ft_success_rate': type_df['ft_success'].mean() * 100,
            'avg_rag_confidence': type_df['rag_confidence'].mean(),
            'avg_ft_confidence': type_df['ft_confidence'].mean(),
            'avg_rag_time': type_df[type_df['rag_time_s'] > 0]['rag_time_s'].mean(),
            'avg_ft_time': type_df[type_df['ft_time_s'] > 0]['ft_time_s'].mean(),
        }
    
    # Overall statistics
    overall_stats = {
        'count': len(df),
        'rag_success_rate': df['rag_success'].mean() * 100,
        'ft_success_rate': df['ft_success'].mean() * 100,
        'avg_rag_confidence': df['rag_confidence'].mean(),
        'avg_ft_confidence': df['ft_confidence'].mean(),
        'avg_rag_time': df[df['rag_time_s'] > 0]['rag_time_s'].mean(),
        'avg_ft_time': df[df['ft_time_s'] > 0]['ft_time_s'].mean(),
    }
    
    return query_type_stats, overall_stats

def analyze_strengths_tradeoffs(df):
    """Analyze strengths and trade-offs of each approach"""
    
    analysis = {
        'rag_strengths': [
            'Provides evidence and source documents for transparency',
            'Handles out-of-domain queries gracefully (returns empty evidence)',
            'More consistent performance across different query types',
            'No training required - can be updated with new documents',
            'Better for factual, numeric queries with specific answers'
        ],
        'rag_weaknesses': [
            'Dependent on quality of document segmentation and indexing',
            'May retrieve irrelevant context for complex queries',
            'Response generation depends on retrieved context quality',
            'Higher latency due to retrieval + generation pipeline'
        ],
        'ft_strengths': [
            'Faster inference once trained (no retrieval step)',
            'Can handle complex, interpretive questions better',
            'Learns domain-specific patterns and terminology',
            'More consistent response style and format',
            'Better for narrative and analytical questions'
        ],
        'ft_weaknesses': [
            'Requires substantial training data and computational resources',
            'May hallucinate or provide incorrect information',
            'Difficult to update with new information without retraining',
            'Performance degrades on out-of-domain queries',
            'Less transparent (no evidence provided)'
        ]
    }
    
    return analysis

def generate_performance_benchmarks(df):
    """Generate performance benchmarking metrics"""
    
    benchmarks = {}
    
    # Accuracy benchmarks
    benchmarks['accuracy'] = {
        'rag_overall': df['rag_success'].mean() * 100,
        'ft_overall': df['ft_success'].mean() * 100,
        'rag_high_confidence': df[df['query_type'] == 'high_confidence']['rag_success'].mean() * 100,
        'ft_high_confidence': df[df['query_type'] == 'high_confidence']['ft_success'].mean() * 100,
        'rag_low_confidence': df[df['query_type'] == 'low_confidence']['rag_success'].mean() * 100,
        'ft_low_confidence': df[df['query_type'] == 'low_confidence']['ft_success'].mean() * 100,
    }
    
    # Latency benchmarks
    benchmarks['latency'] = {
        'rag_avg_time': df[df['rag_time_s'] > 0]['rag_time_s'].mean(),
        'ft_avg_time': df[df['ft_time_s'] > 0]['ft_time_s'].mean(),
        'rag_95th_percentile': df[df['rag_time_s'] > 0]['rag_time_s'].quantile(0.95),
        'ft_95th_percentile': df[df['ft_time_s'] > 0]['ft_time_s'].quantile(0.95),
    }
    
    # Confidence benchmarks
    benchmarks['confidence'] = {
        'rag_avg_confidence': df['rag_confidence'].mean(),
        'ft_avg_confidence': df['ft_confidence'].mean(),
        'rag_confidence_std': df['rag_confidence'].std(),
        'ft_confidence_std': df['ft_confidence'].std(),
    }
    
    return benchmarks

def create_visualizations(df, output_dir):
    """Create visualization charts"""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Success Rate Comparison by Query Type
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Success rates
    query_types = df['query_type'].unique()
    rag_success = [df[df['query_type'] == qt]['rag_success'].mean() * 100 for qt in query_types]
    ft_success = [df[df['query_type'] == qt]['ft_success'].mean() * 100 for qt in query_types]
    
    x = np.arange(len(query_types))
    width = 0.35
    
    axes[0,0].bar(x - width/2, rag_success, width, label='RAG', alpha=0.8)
    axes[0,0].bar(x + width/2, ft_success, width, label='Fine-tuned', alpha=0.8)
    axes[0,0].set_xlabel('Query Type')
    axes[0,0].set_ylabel('Success Rate (%)')
    axes[0,0].set_title('Success Rate Comparison by Query Type')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(query_types, rotation=45)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Confidence scores
    rag_conf = [df[df['query_type'] == qt]['rag_confidence'].mean() for qt in query_types]
    ft_conf = [df[df['query_type'] == qt]['ft_confidence'].mean() for qt in query_types]
    
    axes[0,1].bar(x - width/2, rag_conf, width, label='RAG', alpha=0.8)
    axes[0,1].bar(x + width/2, ft_conf, width, label='Fine-tuned', alpha=0.8)
    axes[0,1].set_xlabel('Query Type')
    axes[0,1].set_ylabel('Average Confidence')
    axes[0,1].set_title('Confidence Score Comparison by Query Type')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(query_types, rotation=45)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Response time comparison
    axes[1,0].boxplot([df[df['rag_time_s'] > 0]['rag_time_s'], 
                       df[df['ft_time_s'] > 0]['ft_time_s']], 
                      labels=['RAG', 'Fine-tuned'])
    axes[1,0].set_ylabel('Response Time (seconds)')
    axes[1,0].set_title('Response Time Distribution')
    axes[1,0].grid(True, alpha=0.3)
    
    # Confidence distribution
    axes[1,1].hist(df['rag_confidence'], alpha=0.7, label='RAG', bins=20)
    axes[1,1].hist(df['ft_confidence'], alpha=0.7, label='Fine-tuned', bins=20)
    axes[1,1].set_xlabel('Confidence Score')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Confidence Score Distribution')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir / 'performance_comparison.png'}")

def generate_recommendations(df, benchmarks):
    """Generate recommendations based on analysis"""
    
    recommendations = {
        'use_rag_when': [
            'Query requires factual, numeric information with specific answers',
            'Transparency and evidence are important',
            'Dealing with out-of-domain or edge case queries',
            'Need to handle new information without retraining',
            'Accuracy is more important than speed'
        ],
        'use_ft_when': [
            'Query requires interpretive or analytical responses',
            'Speed is critical and queries are within domain',
            'Consistent response style and format is needed',
            'Complex narrative questions that benefit from learned patterns',
            'Have sufficient training data and computational resources'
        ],
        'hybrid_approach': [
            'Use RAG for factual queries and FT for interpretive ones',
            'Implement confidence-based routing between approaches',
            'Use FT as primary with RAG fallback for low-confidence responses',
            'Combine both approaches for comprehensive coverage'
        ],
        'improvements': [
            'Enhance RAG with better re-ranking and context selection',
            'Expand FT training data with more diverse question types',
            'Implement adaptive chunking for better document segmentation',
            'Add more sophisticated confidence scoring mechanisms',
            'Develop query classification for automatic approach selection'
        ]
    }
    
    return recommendations

def write_markdown_report(query_type_stats, overall_stats, analysis, benchmarks, recommendations, output_file):
    """Write comprehensive markdown report"""
    
    with open(output_file, 'w') as f:
        f.write("# Comparative Financial QA System: RAG vs Fine-Tuning Analysis Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This report provides a comprehensive analysis of two approaches for financial question answering:\n")
        f.write("1. **Retrieval-Augmented Generation (RAG)** - Combines document retrieval with generative response\n")
        f.write("2. **Fine-Tuned Language Model (FT)** - Directly fine-tunes a language model on financial Q&A\n\n")
        
        f.write("## Performance Comparison Table\n\n")
        f.write("### Overall Performance\n\n")
        f.write("| Metric | RAG | Fine-tuned | Difference |\n")
        f.write("|--------|-----|------------|------------|\n")
        f.write(f"| Success Rate | {overall_stats['rag_success_rate']:.1f}% | {overall_stats['ft_success_rate']:.1f}% | {overall_stats['rag_success_rate'] - overall_stats['ft_success_rate']:+.1f}% |\n")
        f.write(f"| Avg Confidence | {overall_stats['avg_rag_confidence']:.3f} | {overall_stats['avg_ft_confidence']:.3f} | {overall_stats['avg_rag_confidence'] - overall_stats['avg_ft_confidence']:+.3f} |\n")
        f.write(f"| Avg Response Time | {overall_stats['avg_rag_time']:.3f}s | {overall_stats['avg_ft_time']:.3f}s | {overall_stats['avg_rag_time'] - overall_stats['avg_ft_time']:+.3f}s |\n\n")
        
        f.write("### Performance by Query Type\n\n")
        f.write("| Query Type | RAG Success | FT Success | RAG Confidence | FT Confidence |\n")
        f.write("|------------|-------------|------------|----------------|---------------|\n")
        for query_type, stats in query_type_stats.items():
            f.write(f"| {query_type.replace('_', ' ').title()} | {stats['rag_success_rate']:.1f}% | {stats['ft_success_rate']:.1f}% | {stats['avg_rag_confidence']:.3f} | {stats['avg_ft_confidence']:.3f} |\n")
        
        f.write("\n## Detailed Analysis\n\n")
        
        f.write("### RAG System Strengths\n\n")
        for strength in analysis['rag_strengths']:
            f.write(f"- {strength}\n")
        
        f.write("\n### RAG System Weaknesses\n\n")
        for weakness in analysis['rag_weaknesses']:
            f.write(f"- {weakness}\n")
        
        f.write("\n### Fine-tuned System Strengths\n\n")
        for strength in analysis['ft_strengths']:
            f.write(f"- {strength}\n")
        
        f.write("\n### Fine-tuned System Weaknesses\n\n")
        for weakness in analysis['ft_weaknesses']:
            f.write(f"- {weakness}\n")
        
        f.write("\n## Performance Benchmarks\n\n")
        
        f.write("### Accuracy Benchmarks\n\n")
        f.write(f"- **Overall RAG Accuracy**: {benchmarks['accuracy']['rag_overall']:.1f}%\n")
        f.write(f"- **Overall FT Accuracy**: {benchmarks['accuracy']['ft_overall']:.1f}%\n")
        f.write(f"- **High-Confidence Queries**: RAG {benchmarks['accuracy']['rag_high_confidence']:.1f}% vs FT {benchmarks['accuracy']['ft_high_confidence']:.1f}%\n")
        f.write(f"- **Low-Confidence Queries**: RAG {benchmarks['accuracy']['rag_low_confidence']:.1f}% vs FT {benchmarks['accuracy']['ft_low_confidence']:.1f}%\n\n")
        
        f.write("### Latency Benchmarks\n\n")
        f.write(f"- **Average RAG Response Time**: {benchmarks['latency']['rag_avg_time']:.3f}s\n")
        f.write(f"- **Average FT Response Time**: {benchmarks['latency']['ft_avg_time']:.3f}s\n")
        f.write(f"- **95th Percentile RAG**: {benchmarks['latency']['rag_95th_percentile']:.3f}s\n")
        f.write(f"- **95th Percentile FT**: {benchmarks['latency']['ft_95th_percentile']:.3f}s\n\n")
        
        f.write("### Confidence Benchmarks\n\n")
        f.write(f"- **RAG Confidence**: {benchmarks['confidence']['rag_avg_confidence']:.3f} ± {benchmarks['confidence']['rag_confidence_std']:.3f}\n")
        f.write(f"- **FT Confidence**: {benchmarks['confidence']['ft_avg_confidence']:.3f} ± {benchmarks['confidence']['ft_confidence_std']:.3f}\n\n")
        
        f.write("## Recommendations\n\n")
        
        f.write("### When to Use RAG\n\n")
        for rec in recommendations['use_rag_when']:
            f.write(f"- {rec}\n")
        
        f.write("\n### When to Use Fine-tuned Models\n\n")
        for rec in recommendations['use_ft_when']:
            f.write(f"- {rec}\n")
        
        f.write("\n### Hybrid Approaches\n\n")
        for rec in recommendations['hybrid_approach']:
            f.write(f"- {rec}\n")
        
        f.write("\n### Improvement Opportunities\n\n")
        for rec in recommendations['improvements']:
            f.write(f"- {rec}\n")
        
        f.write("\n## Conclusion\n\n")
        f.write("Both RAG and Fine-tuning approaches have distinct advantages for financial question answering:\n\n")
        f.write("- **RAG excels** at factual, evidence-based responses with high transparency\n")
        f.write("- **Fine-tuning excels** at interpretive questions and consistent response generation\n")
        f.write("- **Hybrid approaches** offer the best of both worlds for comprehensive coverage\n\n")
        f.write("The choice between approaches should be based on specific use case requirements, available resources, and performance priorities.\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', required=True, help='Evaluation CSV file')
    ap.add_argument('--out', required=True, help='Output report file')
    ap.add_argument('--plots', action='store_true', help='Generate visualization plots')
    args = ap.parse_args()
    
    print("Loading evaluation data...")
    df = load_evaluation_data(args.eval)
    
    print("Generating comparison table...")
    query_type_stats, overall_stats = generate_comparison_table(df)
    
    print("Analyzing strengths and trade-offs...")
    analysis = analyze_strengths_tradeoffs(df)
    
    print("Generating performance benchmarks...")
    benchmarks = generate_performance_benchmarks(df)
    
    print("Generating recommendations...")
    recommendations = generate_recommendations(df, benchmarks)
    
    print("Writing markdown report...")
    output_path = Path(args.out)
    write_markdown_report(query_type_stats, overall_stats, analysis, benchmarks, recommendations, output_path)
    
    if args.plots:
        print("Generating visualizations...")
        plots_dir = output_path.parent
        create_visualizations(df, plots_dir)
    
    print(f"\nAnalysis report generated successfully!")
    print(f"Report saved to: {args.out}")
    if args.plots:
        print(f"Plots saved to: {plots_dir}")

if __name__ == "__main__":
    main()
