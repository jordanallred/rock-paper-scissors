from traceback import print_tb

from flask import Flask, render_template, request, session, jsonify
import random
import pickle
import base64
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

CHOICES = ['rock', 'paper', 'scissors']

BEATS = {
    'rock': 'scissors',
    'paper': 'rock',
    'scissors': 'paper'
}

COUNTER_TO = {
    'rock': 'paper',
    'paper': 'scissors',
    'scissors': 'rock'
}


class FlexiblePredictor:
    """A flexible predictor that can use any combination of n-gram lengths."""

    def __init__(self, n_gram_range=(1, 5), markov_weights=None):
        """Initialize the predictor with configurable n-gram lengths.

        Args:
            n_gram_range (tuple): The minimum and maximum n-gram lengths to track (inclusive)
            markov_weights (dict, optional): Custom weights for different n-gram lengths
        """
        self.min_n, self.max_n = n_gram_range
        self.n_values = list(range(self.min_n, self.max_n + 1))

        if markov_weights is None:
            self.markov_weights = {n: 0.5 + (0.1 * n) for n in self.n_values}
        else:
            self.markov_weights = markov_weights

        self.history = []
        self.transitions = {n: {} for n in self.n_values}

        self.move_counts = {'rock': 0, 'paper': 0, 'scissors': 0}
        self.last_five_moves = []
        self.last_results = []
        self.switching_patterns = {'win': {}, 'lose': {}, 'tie': {}}

        self.metadata = {
            'total_rounds': 0,
            'streak': {'type': None, 'count': 0},
        }

    def update(self, move, result=None):
        """Update the model with a new move and result.

        Args:
            move (str): The user's move ('rock', 'paper', or 'scissors')
            result (str, optional): The result of the round ('win', 'lose', 'tie')
        """

        self.history.append(move)
        self.move_counts[move] += 1
        self.metadata['total_rounds'] += 1

        self.last_five_moves.append(move)
        if len(self.last_five_moves) > 5:
            self.last_five_moves.pop(0)

        if result:
            if self.metadata['streak']['type'] == result:
                self.metadata['streak']['count'] += 1
            else:
                self.metadata['streak'] = {'type': result, 'count': 1}

        if result is not None and len(self.history) >= 2:
            prev_move = self.history[-2]
            self.last_results.append(result)
            if len(self.last_results) > 5:
                self.last_results.pop(0)

            if prev_move not in self.switching_patterns[result]:
                self.switching_patterns[result][prev_move] = {'rock': 0, 'paper': 0, 'scissors': 0}
            self.switching_patterns[result][prev_move][move] += 1

        for n in self.n_values:
            self._update_ngram(n, move)

    def _update_ngram(self, n, move):
        """Update a specific n-gram transition table.

        Args:
            n (int): The n-gram length
            move (str): The user's move
        """

        if len(self.history) <= n:
            return

        if n == 0:
            context = ()
        else:
            context = tuple(self.history[-(n + 1):-1])

        if context not in self.transitions[n]:
            self.transitions[n][context] = {'rock': 0, 'paper': 0, 'scissors': 0}

        self.transitions[n][context][move] += 1

    def predict(self):
        """Predict the user's next move based on multiple strategies.

        Returns:
            str: Predicted move ('rock', 'paper', or 'scissors')
            float: Confidence in the prediction (0-1)
            str: Strategy used for prediction
        """
        strategies = []

        for n in sorted(self.n_values, reverse=True):
            if len(self.history) >= n:

                if n == 0:
                    context = ()
                else:
                    context = tuple(self.history[-n:])

                if context in self.transitions[n]:
                    counts = self.transitions[n][context]
                    total = sum(counts.values())

                    if total > 0:
                        move_probs = {move: count / total for move, count in counts.items()}
                        predicted_move = max(move_probs, key=move_probs.get)
                        confidence = move_probs[predicted_move]
                        weight = self.markov_weights.get(n, 0.5)
                        strategies.append((predicted_move, confidence, weight, f"N-gram (n={n})"))

        if sum(self.move_counts.values()) > 5:
            total_moves = sum(self.move_counts.values())
            move_probs = {move: count / total_moves for move, count in self.move_counts.items()}
            favorite_move = max(move_probs, key=move_probs.get)
            confidence = move_probs[favorite_move]

            strategies.append((favorite_move, confidence, 0.3, "Favorite move"))

        if len(self.last_results) > 0 and len(self.history) > 0:
            last_result = self.last_results[-1]
            last_move = self.history[-1]

            if last_move in self.switching_patterns[last_result]:
                counts = self.switching_patterns[last_result][last_move]
                total = sum(counts.values())

                if total > 0:
                    move_probs = {move: count / total for move, count in counts.items()}
                    predicted_move = max(move_probs, key=move_probs.get)
                    confidence = move_probs[predicted_move]

                    strategies.append((predicted_move, confidence, 0.7, f"After {last_result}"))

        if self.metadata['streak']['count'] >= 3:
            streak_type = self.metadata['streak']['type']

            if streak_type == 'lose' and len(self.history) > 0:
                last_move = self.history[-1]
                next_move = self._get_move_not_beaten_by(COUNTER_TO[last_move])
                strategies.append((next_move, 0.6, 0.5, f"Breaking {streak_type} streak"))

        if not strategies:
            return random.choice(CHOICES), 0.0, "Random (first move)"

        weighted_votes = {}
        total_weight = 0
        best_strategy = None
        best_confidence = 0

        for move, confidence, weight, strategy in strategies:
            if move not in weighted_votes:
                weighted_votes[move] = 0

            weighted_value = confidence * weight
            weighted_votes[move] += weighted_value
            total_weight += weight

            if confidence > best_confidence:
                best_confidence = confidence
                best_strategy = strategy

        if total_weight > 0:
            for move in weighted_votes:
                weighted_votes[move] /= total_weight

            predicted_move = max(weighted_votes, key=weighted_votes.get)
            confidence = weighted_votes[predicted_move]
            return predicted_move, confidence, best_strategy

        return random.choice(CHOICES), 0.0, "Random"

    def _get_move_not_beaten_by(self, move):
        """Get a move that won't be beaten by the specified move."""
        options = [m for m in CHOICES if COUNTER_TO[m] != move]
        return random.choice(options)

    def get_counter_move(self):
        """Choose a move that counters the predicted user move.

        Returns:
            str: AI's move ('rock', 'paper', or 'scissors')
            float: Confidence in the prediction used for this move (0-1)
            str: Strategy used for prediction
        """
        predicted_move, confidence, strategy = self.predict()

        if confidence < 0.5:
            random_factor = (0.5 - confidence) * 2
            if random.random() < random_factor:
                return random.choice(CHOICES), confidence, "Random (low confidence)"

        return COUNTER_TO.get(predicted_move), confidence, strategy

    def get_transition_heatmap(self, n=None):
        """Get transition data for visualization.

        Args:
            n (int, optional): Specific n-gram length to visualize. If None, uses middle value.

        Returns:
            dict: Transition counts for heatmap visualization
        """
        result = {}

        if n is None:
            n = self.n_values[len(self.n_values) // 2]

        if n not in self.transitions:
            return {}

        for context, next_moves in self.transitions[n].items():
            if len(context) == 0:
                context_str = "Overall"
            else:
                context_str = " → ".join(context)
            result[context_str] = next_moves

        return result

    def get_move_frequencies(self):
        """Get the frequency of each move.

        Returns:
            dict: Frequency counts for each move
        """
        total = sum(self.move_counts.values()) or 1
        return {move: count / total for move, count in self.move_counts.items()}

    def get_strategy_stats(self):
        """Get statistics about which strategies are being used.

        Returns:
            dict: Strategy usage and effectiveness metrics
        """
        stats = {
            "confidence": 0,
            "strategy": "None",
            "patterns": {},
        }

        predicted_move, confidence, strategy = self.predict()
        stats["confidence"] = confidence
        stats["strategy"] = strategy

        top_patterns = []
        for n in self.n_values:
            if len(self.history) >= n:
                if n == 0:
                    context = ()
                else:
                    context = tuple(self.history[-n:])

                if context in self.transitions[n]:
                    counts = self.transitions[n][context]
                    total = sum(counts.values())
                    if total > 0:
                        strongest_move = max(counts, key=counts.get)
                        if n == 0:
                            pattern_str = f"Overall → {strongest_move}"
                        else:
                            pattern_str = " → ".join(context) + f" → {strongest_move}"
                        strength = counts[strongest_move] / total
                        top_patterns.append((pattern_str, strength, total))

        top_patterns.sort(key=lambda x: (x[1], x[2]), reverse=True)
        for i, (pattern, strength, count) in enumerate(top_patterns[:3]):
            stats["patterns"][pattern] = {"strength": strength, "count": count}

        return stats

    def get_markov_analysis(self):
        """Get detailed analysis of the Markov model.

        Returns:
            dict: Detailed Markov model statistics
        """
        model_stats = {}

        for n in self.n_values:
            if not self.transitions[n]:
                continue

            contexts = {}
            for context, next_moves in self.transitions[n].items():
                total = sum(next_moves.values())
                if total > 0:

                    probs = [count / total for count in next_moves.values()]

                    entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)

                    max_prob = max(probs)

                    if len(context) == 0:
                        context_name = "Overall"
                    else:
                        context_name = " → ".join(context)

                    contexts[context_name] = {
                        "entropy": entropy,
                        "predictability": 1.0 - (entropy / np.log2(3)),
                        "max_probability": max_prob,
                        "observations": total
                    }

            model_stats[f"n={n}"] = contexts

        return model_stats


def serialize_predictor(predictor):
    """Serialize the predictor object to store in session."""
    return base64.b64encode(pickle.dumps(predictor)).decode('utf-8')


def deserialize_predictor(serialized_predictor):
    """Deserialize the predictor object from session."""
    try:
        return pickle.loads(base64.b64decode(serialized_predictor.encode('utf-8')))
    except AttributeError:

        print("Creating new predictor due to class mismatch")
        return FlexiblePredictor()


@app.route('/')
def index():
    """Render the game page."""

    if 'predictor' not in session:
        predictor = FlexiblePredictor()
        session['predictor'] = serialize_predictor(predictor)
        session['scores'] = {'user': 0, 'ai': 0, 'tie': 0}
        session['rounds'] = []

    return render_template('index.html',
                           scores=session['scores'],
                           rounds=session['rounds'])


@app.route('/play', methods=['POST'])
def play():
    """Handle a game play."""
    try:
        user_move = request.form.get('move')

        if user_move not in CHOICES:
            return jsonify({'error': 'Invalid move'}), 400

        predictor = deserialize_predictor(session['predictor'])

        ai_move, confidence, strategy = predictor.get_counter_move()

        if user_move == ai_move:
            result = 'tie'
        elif BEATS[user_move] == ai_move:
            result = 'win'
        elif BEATS[ai_move] == user_move:
            result = 'lose'
        else:
            result = 'tie'

        scores = session['scores']
        if result == 'win':
            scores['user'] += 1
        elif result == 'lose':
            scores['ai'] += 1
        else:
            scores['tie'] += 1

        predictor.update(user_move, result)

        rounds = session['rounds']
        rounds.append({
            'user_move': user_move,
            'ai_move': ai_move,
            'result': result,
            'confidence': round(confidence * 100, 1),
            'strategy': strategy
        })
        if len(rounds) > 10:
            rounds.pop(0)

        session['predictor'] = serialize_predictor(predictor)
        session['scores'] = scores
        session['rounds'] = rounds

        try:
            strategy_stats = predictor.get_strategy_stats()
            markov_analysis = predictor.get_markov_analysis()
        except AttributeError:

            strategy_stats = {
                "confidence": confidence,
                "strategy": strategy,
                "patterns": {}
            }
            markov_analysis = {}

        return jsonify({
            'user_move': user_move,
            'ai_move': ai_move,
            'result': result,
            'confidence': round(confidence * 100, 1),
            'strategy': strategy,
            'scores': scores,
            'move_frequencies': predictor.get_move_frequencies(),
            'transitions': predictor.get_transition_heatmap(),
            'strategy_stats': strategy_stats,
            'markov_analysis': markov_analysis
        })
    except Exception as e:
        print_tb(e.__traceback__)
        print(f"Error in play route: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset():
    """Reset the game."""

    predictor = FlexiblePredictor()
    session['predictor'] = serialize_predictor(predictor)
    session['scores'] = {'user': 0, 'ai': 0, 'tie': 0}
    session['rounds'] = []

    return jsonify({'message': 'Game reset successfully'})

if __name__ == '__main__':
    app.run(host="0.0.0.0")
