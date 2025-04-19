const moveIcons = {
    'rock': '✊',
    'paper': '✋',
    'scissors': '✌️'
};

let moveChart = null;


const moveData = {
    labels: ['Rock', 'Paper', 'Scissors'],
    datasets: [{
        label: 'Move Frequency',
        data: [0, 0, 0],
        backgroundColor: [
            'rgba(54, 162, 235, 0.6)',
            'rgba(75, 192, 192, 0.6)',
            'rgba(255, 99, 132, 0.6)'
        ],
        borderColor: [
            'rgba(54, 162, 235, 1)',
            'rgba(75, 192, 192, 1)',
            'rgba(255, 99, 132, 1)'
        ],
        borderWidth: 1
    }]
};


document.addEventListener('DOMContentLoaded', function () {
    initGame();
    addEventListeners();
});


function initGame() {

    try {
        initMoveChart();
    } catch (err) {
        console.warn("Could not initialize chart:", err);
        document.getElementById('moveChart').innerHTML =
            '<p>Chart visualization not available</p>';
    }


    document.getElementById('heatmapContainer').innerHTML =
        '<p>Not enough data to display patterns yet.</p>';
}


function addEventListeners() {

    document.querySelectorAll('.move-button').forEach(button => {
        button.addEventListener('click', function () {
            const move = this.getAttribute('data-move');
            playMove(move);
        });
    });


    document.getElementById('resetButton').addEventListener('click', function () {
        resetGame();
    });
}


function initMoveChart() {
    if (typeof Chart === 'undefined') {
        throw new Error("Chart.js library not available");
    }

    const ctx = document.getElementById('moveChart').getContext('2d');
    moveChart = new Chart(ctx, {
        type: 'bar',
        data: moveData,
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function (value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return context.raw.toFixed(1) + '%';
                        }
                    }
                }
            }
        }
    });
}


function updateMoveChart(frequencies) {
    try {
        if (!moveChart) {
            initMoveChart();
        }

        moveChart.data.datasets[0].data = [
            frequencies.rock * 100 || 0,
            frequencies.paper * 100 || 0,
            frequencies.scissors * 100 || 0
        ];
        moveChart.update();
    } catch (err) {
        console.warn("Could not update chart:", err);
    }
}


function updateHeatmap(transitions) {
    const container = document.getElementById('heatmapContainer');
    container.innerHTML = '';

    if (!transitions || Object.keys(transitions).length === 0) {
        container.innerHTML = '<p>Not enough data to display patterns yet.</p>';
        return;
    }


    let table = document.createElement('table');
    table.style.width = '100%';
    table.style.borderCollapse = 'collapse';


    let thead = document.createElement('thead');
    let headerRow = document.createElement('tr');

    let cornerCell = document.createElement('th');
    cornerCell.textContent = 'Pattern → Next';
    cornerCell.style.padding = '8px';
    cornerCell.style.borderBottom = '1px solid #ddd';
    headerRow.appendChild(cornerCell);

    ['Rock', 'Paper', 'Scissors'].forEach(move => {
        let th = document.createElement('th');
        th.textContent = move;
        th.style.padding = '8px';
        th.style.borderBottom = '1px solid #ddd';
        headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);
    table.appendChild(thead);


    let tbody = document.createElement('tbody');

    Object.entries(transitions).forEach(([pattern, counts]) => {
        let row = document.createElement('tr');


        let labelCell = document.createElement('td');
        labelCell.textContent = pattern;
        labelCell.style.padding = '8px';
        labelCell.style.borderBottom = '1px solid #ddd';
        row.appendChild(labelCell);


        const total = counts.rock + counts.paper + counts.scissors || 0;


        ['rock', 'paper', 'scissors'].forEach(move => {
            let cell = document.createElement('td');
            const count = counts[move] || 0;
            const percentage = total ? (count / total * 100).toFixed(1) : 0;


            const intensity = Math.min(0.9, count / (total || 1));
            const backgroundColor = move === 'rock'
                ? `rgba(54, 162, 235, ${intensity})`
                : move === 'paper'
                    ? `rgba(75, 192, 192, ${intensity})`
                    : `rgba(255, 99, 132, ${intensity})`;

            cell.textContent = `${count} (${percentage}%)`;
            cell.style.padding = '8px';
            cell.style.textAlign = 'center';
            cell.style.backgroundColor = backgroundColor;
            cell.style.color = intensity > 0.6 ? 'white' : 'black';
            cell.style.borderBottom = '1px solid #ddd';
            row.appendChild(cell);
        });

        tbody.appendChild(row);
    });

    table.appendChild(tbody);
    container.appendChild(table);
}

function updateHistory(data) {
    const historyContainer = document.getElementById('historyContainer');

    if (historyContainer.children.length === 1 &&
        historyContainer.firstChild.textContent.includes('No rounds played yet')) {
        historyContainer.innerHTML = '';
    }

    const roundNumber = historyContainer.children.length + 1;

    const historyItem = document.createElement('div');
    historyItem.className = 'history-item';

    const moveText = document.createElement('span');
    moveText.textContent = `Round ${roundNumber}: You played ${data.user_move}, AI played ${data.ai_move} `;

    const resultText = document.createElement('span');
    resultText.className = `result-${data.result}`;
    if (data.result === 'win') {
        resultText.textContent = 'You won!';
    } else if (data.result === 'lose') {
        resultText.textContent = 'AI won!';
    } else {
        resultText.textContent = 'Tie!';
    }

    historyItem.appendChild(moveText);
    historyItem.appendChild(resultText);

    if (data.strategy) {
        const strategyText = document.createElement('div');
        strategyText.textContent = `AI strategy: ${data.strategy}`;
        strategyText.style.fontSize = '0.8em';
        strategyText.style.color = '#666';
        historyItem.appendChild(strategyText);
    }

    historyContainer.appendChild(historyItem);

    while (historyContainer.children.length > 10) {
        historyContainer.removeChild(historyContainer.firstChild);
    }

    historyContainer.scrollTop = historyContainer.scrollHeight;
}

function playMove(move) {
    const formData = new FormData();
    formData.append('move', move);

    fetch('/play', {
        method: 'POST',
        body: formData
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            try {

                document.getElementById('userMoveIcon').textContent = moveIcons[data.user_move];
                document.getElementById('aiMoveIcon').textContent = moveIcons[data.ai_move];


                let resultText = '';
                let resultClass = '';
                if (data.result === 'win') {
                    resultText = 'You won!';
                    resultClass = 'result-win';
                } else if (data.result === 'lose') {
                    resultText = 'AI won!';
                    resultClass = 'result-lose';
                } else {
                    resultText = 'It\'s a tie!';
                    resultClass = 'result-tie';
                }
                document.getElementById('resultDisplay').textContent = resultText;
                document.getElementById('resultDisplay').className = resultClass;


                document.getElementById('confidenceDisplay').textContent =
                    `AI confidence: ${data.confidence}%`;

                if (data.strategy) {
                    document.getElementById('strategyDisplay').textContent =
                        `Strategy: ${data.strategy}`;
                }


                document.getElementById('userScore').textContent = data.scores.user;
                document.getElementById('aiScore').textContent = data.scores.ai;
                document.getElementById('tieScore').textContent = data.scores.tie;


                updateHistory(data);


                if (data.move_frequencies) {
                    updateMoveChart(data.move_frequencies);
                }

                if (data.transitions) {
                    updateHeatmap(data.transitions);
                }

                if (data.strategy_stats) {
                    updateStrategyAnalysis(data.strategy_stats);
                }

            } catch (err) {
                console.warn("Error updating UI:", err);

            }
        })
        .catch(err => {
            console.error("Error playing move:", err);

            if (document.getElementById('userMoveIcon').textContent === '?' ||
                document.getElementById('aiMoveIcon').textContent === '?') {
                alert('Error playing move. Please try again.');
            }
        });
}


function resetGame() {
    fetch('/reset', {
        method: 'POST'
    })
        .then(response => response.json())
        .then(data => {

            document.getElementById('userMoveIcon').textContent = '?';
            document.getElementById('aiMoveIcon').textContent = '?';
            document.getElementById('resultDisplay').textContent = 'Choose your move!';
            document.getElementById('resultDisplay').className = '';
            document.getElementById('confidenceDisplay').textContent = '';
            document.getElementById('strategyDisplay').textContent = '';


            document.getElementById('userScore').textContent = '0';
            document.getElementById('aiScore').textContent = '0';
            document.getElementById('tieScore').textContent = '0';


            document.getElementById('historyContainer').innerHTML =
                '<div class="history-item">No rounds played yet.</div>';


            try {
                if (moveChart) {
                    moveChart.data.datasets[0].data = [0, 0, 0];
                    moveChart.update();
                }
            } catch (err) {
                console.warn("Error resetting chart:", err);
            }

            document.getElementById('heatmapContainer').innerHTML =
                '<p>Not enough data to display patterns yet.</p>';

            alert('Game has been reset!');
        })
        .catch(err => {
            console.error("Error resetting game:", err);
            alert('Error resetting game. Please try again.');
        });
}
