// static/js/models_chart.js
function renderMetricsChart(selectedModel, metrics) {
    const labels = ["RMSE", "MAE", "Median AE", "Explained Variance"];

    const trainData = [
        
        metrics.Train.RMSE,
        metrics.Train.MAE,
        metrics.Train.MedianAE,
        metrics.Train.ExplainedVariance
    ];
    const testData = [
        
        metrics.Test.RMSE,
        metrics.Test.MAE,
        metrics.Test.MedianAE,
        metrics.Test.ExplainedVariance
    ];

    const data = {
        labels: labels,
        datasets: [
            {
                label: 'Train',
                data: trainData,
                backgroundColor: 'rgba(54, 162, 235, 0.6)'
            },
            {
                label: 'Test',
                data: testData,
                backgroundColor: 'rgba(255, 99, 132, 0.6)'
            }
        ]
    };

    const config = {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' },
                title: { display: true, text: selectedModel + ' Metrics Comparison' }
            }
        }
    };

    new Chart(document.getElementById('metricsChart'), config);
}
