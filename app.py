from flask import Flask, render_template, request, session, jsonify
import random
import pickle
import base64
import numpy as np
import os
import tempfile
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = tempfile.mkdtemp()
app.config['SESSION_PERMANENT'] = False

try:
    from flask_session import Session

    Session(app)
except ImportError:
    print("WARNING: Flask-Session not installed. Large sessions will fail.")
    print("Install with: pip install Flask-Session")

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


class ImprovedMLPredictor:
    """Enhanced predictor that exploits human psychological patterns in RPS."""

    def __init__(self, n_gram_range=(1, 7), confidence_threshold=0.48):
        """Initialize the predictor with anti-human psychology strategies.

        Args:
            n_gram_range (tuple): The minimum and maximum n-gram lengths to track (inclusive)
            confidence_threshold (float): Threshold to determine when to trust ML vs exploit human biases
        """
        self.min_n, self.max_n = n_gram_range
        self.n_values = list(range(self.min_n, self.max_n + 1))
        self.confidence_threshold = confidence_threshold

        self.ngram_weights = {n: n ** 1.5 for n in self.n_values}

        self.history = []
        self.transitions = {n: {} for n in self.n_values}
        self.move_counts = {'rock': 0, 'paper': 0, 'scissors': 0}
        self.ai_moves = []
        self.results = []

        self.prediction_accuracy = {'correct': 0, 'total': 0}
        self.randomness_used = 0
        self.total_decisions = 0
        
        # Anti-human psychology tracking
        self.streak_lengths = {'rock': 0, 'paper': 0, 'scissors': 0}
        self.current_streak = {'move': None, 'count': 0}
        self.win_loss_streak = {'type': None, 'count': 0}
        self.rotation_patterns = []
        self.gambler_fallacy_count = 0

    def update(self, move, result=None, ai_move=None):
        """Update the model with a new move and result.

        Args:
            move (str): The user's move ('rock', 'paper', or 'scissors')
            result (str, optional): The result of the round ('win', 'lose', 'tie')
            ai_move (str, optional): The AI's move for this round
        """
        self.history.append(move)
        self.move_counts[move] += 1
        
        if ai_move:
            self.ai_moves.append(ai_move)
        if result:
            self.results.append(result)

        # Update streak tracking
        self._update_streaks(move, result)
        
        # Track rotation patterns (rock->paper->scissors cycles)
        self._update_rotation_patterns(move)

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

        context = tuple(self.history[-(n + 1):-1]) if n > 0 else ()

        if context not in self.transitions[n]:
            self.transitions[n][context] = {'rock': 0, 'paper': 0, 'scissors': 0}

        self.transitions[n][context][move] += 1

    def _get_next_move_distribution(self, n, context):
        """Get the probability distribution for the next move given a context.

        Args:
            n (int): The n-gram length
            context (tuple): The context to check

        Returns:
            dict: Probability distribution for next move
            float: Total count of observations
        """
        if context not in self.transitions[n]:
            return None, 0

        counts = self.transitions[n][context]
        total = sum(counts.values())

        if total == 0:
            return None, 0

        return {move: count / total for move, count in counts.items()}, total

    def predict(self):
        """Predict the user's next move based purely on observed patterns.

        Returns:
            str: Predicted move ('rock', 'paper', or 'scissors')
            float: Confidence in the prediction (0-1)
            str: Strategy used for prediction
        """

        predictions = []

        for n in sorted(self.n_values, reverse=True):
            if len(self.history) < n:
                continue

            context = tuple(self.history[-n:]) if n > 0 else ()

            distribution, observations = self._get_next_move_distribution(n, context)

            if distribution:
                predicted_move = max(distribution, key=distribution.get)
                confidence = distribution[predicted_move]
                weight = self.ngram_weights.get(n, 1.0) * (min(observations, 10) / 10)

                predictions.append({
                    'move': predicted_move,
                    'confidence': confidence,
                    'weight': weight,
                    'strategy': f"{n}-gram pattern" if n > 0 else "Base frequency",
                    'observations': observations
                })

        if not predictions:
            return random.choice(CHOICES), 0.0, "Random (insufficient data)"

        best_prediction = max(predictions, key=lambda x: x['confidence'] * x['weight'])

        return best_prediction['move'], best_prediction['confidence'], best_prediction['strategy']

    def get_counter_move(self):
        """Choose a move using advanced anti-human strategies.

        Returns:
            str: AI's move ('rock', 'paper', or 'scissors')
            float: Confidence in the prediction (0-1)
            str: Strategy used
        """
        self.total_decisions += 1

        # Try psychological exploitation strategies first
        psych_move, psych_confidence, psych_strategy = self._exploit_human_psychology()
        if psych_confidence > 0.7:
            return psych_move, psych_confidence, psych_strategy

        # Enhanced pattern prediction with anti-pattern detection
        predicted_move, confidence, strategy = self._enhanced_predict()

        # Use adaptive confidence threshold based on game state
        adaptive_threshold = self._get_adaptive_threshold()
        
        if confidence < adaptive_threshold:
            # Instead of pure random, use frequency exploitation
            exploit_move, exploit_confidence = self._exploit_frequency_bias()
            if exploit_confidence > 0.4:
                return exploit_move, exploit_confidence, "Frequency exploitation"
            
            self.randomness_used += 1
            return random.choice(CHOICES), confidence, "Random (low confidence)"

        return COUNTER_TO[predicted_move], confidence, strategy

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
            "ml_stats": {
                "randomness_ratio": round(self.randomness_used / max(1, self.total_decisions), 3),
                "longest_effective_ngram": 0
            }
        }

        predicted_move, confidence, strategy = self.predict()
        stats["confidence"] = confidence
        stats["strategy"] = strategy

        effective_ngrams = []
        for n in self.n_values:
            if len(self.history) >= n:
                context = tuple(self.history[-n:]) if n > 0 else ()
                distribution, observations = self._get_next_move_distribution(n, context)

                if distribution and observations >= 3:
                    max_prob = max(distribution.values())
                    if max_prob >= self.confidence_threshold:
                        effective_ngrams.append((n, max_prob, observations))

        if effective_ngrams:
            stats["ml_stats"]["longest_effective_ngram"] = max(effective_ngrams, key=lambda x: x[0])[0]

        pattern_strengths = []
        for n in self.n_values:
            if len(self.history) >= n:
                context = tuple(self.history[-n:]) if n > 0 else ()
                distribution, observations = self._get_next_move_distribution(n, context)

                if distribution and observations >= 2:
                    strongest_move = max(distribution, key=distribution.get)
                    if n == 0:
                        pattern_str = f"Overall → {strongest_move}"
                    else:
                        pattern_str = " → ".join(context) + f" → {strongest_move}"
                    strength = distribution[strongest_move]
                    pattern_strengths.append((pattern_str, strength, observations))

        pattern_strengths.sort(key=lambda x: (x[1], x[2]), reverse=True)
        for index, (pattern, strength, count) in enumerate(pattern_strengths[:5]):
            stats["patterns"][pattern] = {"strength": strength, "count": count}

        return stats

    def get_ml_insights(self):
        """Get insights into how the ML model is performing.

        Returns:
            dict: ML model insights
        """
        insights = {
            "confidence_by_ngram": {},
            "effective_ngrams": [],
            "randomness_rate": self.randomness_used / max(1, self.total_decisions),
            "pattern_entropy": {}
        }

        for n in self.n_values:
            if n not in self.transitions or not self.transitions[n]:
                continue

            confidences = []
            for context, counts in self.transitions[n].items():
                total = sum(counts.values())
                if total >= 3:
                    probs = [count / total for count in counts.values()]
                    max_prob = max(probs)
                    confidences.append(max_prob)

            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                insights["confidence_by_ngram"][n] = avg_confidence

                if avg_confidence >= self.confidence_threshold:
                    insights["effective_ngrams"].append(n)

            for context, counts in self.transitions[n].items():
                total = sum(counts.values())
                if total >= 3:
                    probs = [count / total for count in counts.values()]
                    entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)

                    context_str = "Overall" if not context else " → ".join(context)
                    insights["pattern_entropy"][context_str] = {
                        "entropy": entropy,
                        "predictability": 1.0 - (entropy / np.log2(3)),
                        "observations": total
                    }

        return insights

    def _update_streaks(self, move, result):
        """Track streaks and win/loss patterns."""
        # Update current move streak
        if self.current_streak['move'] == move:
            self.current_streak['count'] += 1
        else:
            self.current_streak = {'move': move, 'count': 1}
            
        # Update win/loss streak
        if result:
            if self.win_loss_streak['type'] == result:
                self.win_loss_streak['count'] += 1
            else:
                self.win_loss_streak = {'type': result, 'count': 1}

    def _update_rotation_patterns(self, move):
        """Track if player follows rock->paper->scissors rotation patterns."""
        if len(self.history) >= 3:
            last_three = self.history[-3:]
            if (last_three == ['rock', 'paper', 'scissors'] or 
                last_three == ['scissors', 'rock', 'paper'] or
                last_three == ['paper', 'scissors', 'rock']):
                self.rotation_patterns.append(len(self.history))

    def _exploit_human_psychology(self):
        """Exploit common human psychological patterns."""
        if len(self.history) < 3:
            return None, 0.0, ""
            
        # Anti-streak: Humans often break streaks after 3+ moves
        if self.current_streak['count'] >= 3:
            # Predict they'll switch to the next move in RPS cycle
            next_in_cycle = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
            predicted = next_in_cycle[self.current_streak['move']]
            return COUNTER_TO[predicted], 0.75, f"Anti-streak after {self.current_streak['count']} {self.current_streak['move']}"
            
        # Gambler's fallacy: After losing, humans often repeat the "winning" move
        if len(self.results) >= 2 and self.results[-1] == 'lose' and self.results[-2] == 'lose':
            # They might repeat the move that would have won last round
            last_ai_move = self.ai_moves[-1] if self.ai_moves else None
            if last_ai_move:
                winning_move = COUNTER_TO[last_ai_move]
                return COUNTER_TO[winning_move], 0.65, "Anti-gambler's fallacy"
                
        # Anti-rotation: Detect and counter RPS rotation patterns
        if len(self.rotation_patterns) >= 2:
            recent_rotations = [p for p in self.rotation_patterns if len(self.history) - p < 10]
            if len(recent_rotations) >= 2:
                # Predict next in rotation
                current = self.history[-1]
                next_rotation = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
                predicted = next_rotation[current]
                return COUNTER_TO[predicted], 0.7, "Anti-rotation pattern"
                
        # Win-stay, lose-shift psychology
        if len(self.results) >= 1:
            if self.results[-1] == 'win':  # They won, might repeat
                return COUNTER_TO[self.history[-1]], 0.6, "Counter win-stay tendency"
            elif self.results[-1] == 'lose':  # They lost, might shift
                # Predict they'll avoid their last move
                avoided_move = self.history[-1]
                likely_moves = [m for m in CHOICES if m != avoided_move]
                predicted = random.choice(likely_moves)
                return COUNTER_TO[predicted], 0.55, "Counter lose-shift tendency"
                
        return None, 0.0, ""

    def _enhanced_predict(self):
        """Enhanced prediction with anti-pattern detection."""
        predictions = []
        
        # Standard n-gram predictions
        for n in sorted(self.n_values, reverse=True):
            if len(self.history) < n:
                continue
                
            context = tuple(self.history[-n:]) if n > 0 else ()
            distribution, observations = self._get_next_move_distribution(n, context)
            
            if distribution:
                predicted_move = max(distribution, key=distribution.get)
                confidence = distribution[predicted_move]
                
                # Boost confidence for patterns with clear dominance
                entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in distribution.values())
                max_entropy = np.log2(3)  # Maximum entropy for 3 choices
                clarity_boost = 1 - (entropy / max_entropy)
                confidence = confidence * (1 + clarity_boost * 0.3)
                
                weight = self.ngram_weights.get(n, 1.0) * (min(observations, 15) / 15)
                
                predictions.append({
                    'move': predicted_move,
                    'confidence': min(confidence, 1.0),
                    'weight': weight,
                    'strategy': f"Enhanced {n}-gram pattern" if n > 0 else "Enhanced frequency",
                    'observations': observations
                })
        
        if not predictions:
            return random.choice(CHOICES), 0.0, "Random (insufficient data)"
            
        best_prediction = max(predictions, key=lambda x: x['confidence'] * x['weight'])
        return best_prediction['move'], best_prediction['confidence'], best_prediction['strategy']

    def _get_adaptive_threshold(self):
        """Get adaptive confidence threshold based on game state."""
        base_threshold = self.confidence_threshold
        
        # Lower threshold when we're losing (be more aggressive)
        if len(self.results) >= 5:
            recent_results = self.results[-5:]
            win_rate = recent_results.count('win') / len(recent_results)
            if win_rate < 0.4:
                base_threshold *= 0.85
            elif win_rate > 0.6:
                base_threshold *= 1.1
                
        # Lower threshold early in game when we have less data
        if len(self.history) < 10:
            base_threshold *= 0.9
            
        return min(max(base_threshold, 0.3), 0.8)

    def _exploit_frequency_bias(self):
        """Exploit human frequency biases (Rock bias, etc.)."""
        if len(self.history) < 5:
            # Exploit rock bias in early game
            return COUNTER_TO['rock'], 0.5
            
        frequencies = self.get_move_frequencies()
        
        # Find least used move (humans avoid it due to psychological reasons)
        least_used = min(frequencies.keys(), key=lambda k: frequencies[k])
        most_used = max(frequencies.keys(), key=lambda k: frequencies[k])
        
        # If there's a strong bias, exploit it
        if frequencies[most_used] - frequencies[least_used] > 0.15:
            return COUNTER_TO[most_used], 0.55
            
        # Default to countering rock (most common human choice)
        return COUNTER_TO['rock'], 0.45


def get_predictor_path(session_id):
    """Get the path for storing predictor data."""
    predictor_dir = os.path.join(tempfile.gettempdir(), 'rps_predictors')
    os.makedirs(predictor_dir, exist_ok=True)
    return os.path.join(predictor_dir, f"predictor_{session_id}.pkl")


def save_predictor(predictor, session_id):
    """Save predictor to filesystem instead of session."""
    path = get_predictor_path(session_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(predictor, f)
    return path


def load_predictor(session_id):
    """Load predictor from filesystem."""
    path = get_predictor_path(session_id)
    try:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
    except (pickle.UnpicklingError, EOFError):
        if os.path.exists(path):
            os.remove(path)
    return ImprovedMLPredictor()


@app.route('/')
def index():
    """Render the game page."""

    if 'id' not in session:
        session['id'] = base64.b64encode(os.urandom(16)).decode('utf-8')

    predictor = load_predictor(session['id'])

    if 'scores' not in session:
        session['scores'] = {'user': 0, 'ai': 0, 'tie': 0}
    if 'round_number' not in session:
        session['round_number'] = 0

    return render_template('index.html',
                           scores=session['scores'])


@app.route('/play', methods=['POST'])
def play():
    """Handle a game play."""
    try:
        user_move = request.form.get('move')

        if user_move not in CHOICES:
            return jsonify({'error': 'Invalid move'}), 400

        if 'id' not in session:
            session['id'] = base64.b64encode(os.urandom(16)).decode('utf-8')

        predictor = load_predictor(session['id'])

        ai_move, confidence, strategy = predictor.get_counter_move()

        if user_move == ai_move:
            result = 'tie'
        elif BEATS[user_move] == ai_move:
            result = 'win'
        elif BEATS[ai_move] == user_move:
            result = 'lose'
        else:
            result = 'tie'

        if 'scores' not in session:
            session['scores'] = {'user': 0, 'ai': 0, 'tie': 0}

        scores = session['scores']
        if result == 'win':
            scores['user'] += 1
        elif result == 'lose':
            scores['ai'] += 1
        else:
            scores['tie'] += 1

        predictor.update(user_move, result, ai_move)

        save_predictor(predictor, session['id'])

        session['round_number'] += 1
        total_round_number = session['round_number']

        new_round = {
            'round_number': total_round_number,
            'user_move': user_move,
            'ai_move': ai_move,
            'result': result,
            'confidence': round(confidence * 100, 1),
            'strategy': strategy
        }

        session['scores'] = scores

        try:
            strategy_stats = predictor.get_strategy_stats()
            ml_insights = predictor.get_ml_insights()
        except AttributeError:
            strategy_stats = {
                "confidence": confidence,
                "strategy": strategy,
                "patterns": {}
            }
            ml_insights = {}

        return jsonify({
            'round_number': new_round['round_number'],
            'user_move': user_move,
            'ai_move': ai_move,
            'result': result,
            'confidence': round(confidence * 100, 1),
            'strategy': strategy,
            'scores': scores,
            'move_frequencies': predictor.get_move_frequencies(),
            'transitions': predictor.get_transition_heatmap(),
            'strategy_stats': strategy_stats,
            'ml_insights': ml_insights
        })
    except Exception as e:
        print(f"Error in play route: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset():
    """Reset the game."""

    predictor = ImprovedMLPredictor()

    if 'id' not in session:
        session['id'] = base64.b64encode(os.urandom(16)).decode('utf-8')

    save_predictor(predictor, session['id'])

    session['scores'] = {'user': 0, 'ai': 0, 'tie': 0}
    session['round_number'] = 0

    return jsonify({'message': 'Game reset successfully'})


@app.before_request
def cleanup_old_predictors():
    """Periodically clean up old predictor files."""
    if random.random() < 0.05:
        predictor_dir = os.path.join(tempfile.gettempdir(), 'rps_predictors')
        if os.path.exists(predictor_dir):
            current_time = time.time()
            for filename in os.listdir(predictor_dir):
                filepath = os.path.join(predictor_dir, filename)

                if os.path.isfile(filepath) and current_time - os.path.getmtime(filepath) > 86400:
                    try:
                        os.remove(filepath)
                    except:
                        pass


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
