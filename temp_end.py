    return {
        'experiment_id': results.experiment_id,
        'transferability': results.evaluation_summary.get('cross_model_performance', {}).get('avg_transferability', 0.0),
        'conclusions': results.conclusions.get('transferability_assessment', 'Unknown')
    }