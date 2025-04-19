from flask import Flask, render_template, request, session, jsonify
import random
import pickle
import base64
from collections import defaultdict

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure random key in production

# Game choices
CHOICES = ['rock', 'paper', 'scissors']
# Winning combinations: key beats value
BEATS = {
    'rock': 'scissors',
    'paper': 'rock',
    'scissors': 'paper'
}
# Counter moves: to beat key, play value
COUNTER_TO = {
    'rock': 'paper',
    'paper': 'scissors',
    'scissors': 'rock'
}


class EnhancedPredictor:
    def __init__(self, n_values=[2, 3, 4]):
        """Initialize an enhanced predictor for RPS game.

        Args:
            n_values (list): Different context lengths to track for prediction.
        """
        self.n_values = n_values
        self.history = []  # Store user's move history
        self.transitions = {n: {} for n in n_values}  # Store transition counts for each n-value

        # Track additional patterns
        self.move_counts = {'rock': 0, 'paper': 0, 'scissors': 0}
        self.last_five_moves = []
        self.last_results = []  # Track win/lose/tie results
        self.switching_patterns = {'win': {}, 'lose': {}, 'tie': {}}  # Track move changes after results

    def update(self, move, result=None):
        """Update the model with a new move and result.

        Args:
            move (str): The user's move ('rock', 'paper', or 'scissors')
            result (str, optional): The result of the round ('win', 'lose', 'tie')
        """
        # Update basic history
        self.history.append(move)
        self.move_counts[move] += 1

        # Update last five moves
        self.last_five_moves.append(move)
        if len(self.last_five_moves) > 5:
            self.last_five_moves.pop(0)

        # Update result-based patterns if result is provided
        if result is not None and len(self.history) >= 2:
            prev_move = self.history[-2]
            self.last_results.append(result)
            if len(self.last_results) > 5:
                self.last_results.pop(0)

            # Track switching patterns after wins/losses/ties
            if prev_move not in self.switching_patterns[result]:
                self.switching_patterns[result][prev_move] = {'rock': 0, 'paper': 0, 'scissors': 0}
            self.switching_patterns[result][prev_move][move] += 1

        # Update n-gram transitions for each n value
        for n in self.n_values:
            if len(self.history) > n:
                # Get the n-gram context
                context = tuple(self.history[-(n + 1):-1])

                # Initialize nested dictionary if needed
                if context not in self.transitions[n]:
                    self.transitions[n][context] = {'rock': 0, 'paper': 0, 'scissors': 0}

                # Increment the count for this context -> move transition
                self.transitions[n][context][move] += 1

    def predict(self):
        """Predict the user's next move based on multiple strategies.

        Returns:
            str: Predicted move ('rock', 'paper', or 'scissors')
            float: Confidence in the prediction (0-1)
            str: Strategy used for prediction
        """
        strategies = []

        # Strategy 1: Most recent N-gram patterns (weighted by recency)
        for n in sorted(self.n_values, reverse=True):  # Prefer longer patterns
            if len(self.history) >= n:
                # Get the current context
                context = tuple(self.history[-n:])

                # Check if we've seen this context before
                if context in self.transitions[n]:
                    counts = self.transitions[n][context]
                    total = sum(counts.values())

                    if total > 0:
                        move_probs = {move: count / total for move, count in counts.items()}
                        predicted_move = max(move_probs, key=move_probs.get)
                        confidence = move_probs[predicted_move]
                        weight = 0.5 + (0.1 * n)  # Higher weight for longer patterns
                        strategies.append((predicted_move, confidence, weight, f"N-gram (n={n})"))

        # Strategy 2: Favorite move (frequency-based)
        if sum(self.move_counts.values()) > 5:
            total_moves = sum(self.move_counts.values())
            move_probs = {move: count / total_moves for move, count in self.move_counts.items()}
            favorite_move = max(move_probs, key=move_probs.get)
            confidence = move_probs[favorite_move]
            # Lower weight for this simpler strategy
            strategies.append((favorite_move, confidence, 0.3, "Favorite move"))

        # Strategy 3: Result-based switching pattern
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
                    # Higher weight because this is a strong psychological pattern
                    strategies.append((predicted_move, confidence, 0.7, f"After {last_result}"))

        # If we have no strategies, return a random move
        if not strategies:
            return random.choice(CHOICES), 0.0, "Random (first move)"

        # Combine strategies with weighting
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

            # Track the strategy with highest confidence
            if confidence > best_confidence:
                best_confidence = confidence
                best_strategy = strategy

        # Normalize and select top move
        if total_weight > 0:
            for move in weighted_votes:
                weighted_votes[move] /= total_weight

            predicted_move = max(weighted_votes, key=weighted_votes.get)
            confidence = weighted_votes[predicted_move]
            return predicted_move, confidence, best_strategy

        # Fallback to random
        return random.choice(CHOICES), 0.0, "Random"

    def get_counter_move(self):
        """Choose a move that counters the predicted user move.

        Returns:
            str: AI's move ('rock', 'paper', or 'scissors')
            float: Confidence in the prediction used for this move (0-1)
            str: Strategy used for prediction
        """
        predicted_move, confidence, strategy = self.predict()

        # Add some randomness based on confidence
        if confidence < 0.5:
            # Lower confidence = more randomness
            random_factor = (0.5 - confidence) * 2  # 0 to 1 scale
            if random.random() < random_factor:
                return random.choice(CHOICES), confidence, "Random (low confidence)"
        elif confidence > 0.8:
            # Very high confidence - occasionally be unpredictable
            if random.random() < 0.1:  # 10% of the time be unpredictable
                counter_choices = [move for move in CHOICES if move != COUNTER_TO.get(predicted_move)]
                return random.choice(counter_choices), confidence, "Deception (high confidence)"

        # Counter the predicted move (use the move that beats the predicted move)
        return COUNTER_TO.get(predicted_move), confidence, strategy

    def get_transition_heatmap(self):
        """Get transition data for visualization.

        Returns:
            dict: Transition counts for heatmap visualization
        """
        result = {}

        # Use the middle n-value for visualization
        middle_n = self.n_values[len(self.n_values) // 2]

        for context, next_moves in self.transitions[middle_n].items():
            context_str = " → ".join(context)
            result[context_str] = next_moves

        return result

    def get_move_frequencies(self):
        """Get the frequency of each move.

        Returns:
            dict: Frequency counts for each move
        """
        total = sum(self.move_counts.values()) or 1  # Avoid division by zero
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

        # Get the current prediction
        predicted_move, confidence, strategy = self.predict()
        stats["confidence"] = confidence
        stats["strategy"] = strategy

        # Get pattern strength for top contexts
        top_patterns = []
        for n in self.n_values:
            if len(self.history) >= n:
                context = tuple(self.history[-n:])
                if context in self.transitions[n]:
                    counts = self.transitions[n][context]
                    total = sum(counts.values())
                    if total > 0:
                        strongest_move = max(counts, key=counts.get)
                        pattern_str = " → ".join(context) + f" → {strongest_move}"
                        strength = counts[strongest_move] / total
                        top_patterns.append((pattern_str, strength, total))

        # Sort by strength and take top 3
        top_patterns.sort(key=lambda x: (x[1], x[2]), reverse=True)
        for i, (pattern, strength, count) in enumerate(top_patterns[:3]):
            stats["patterns"][pattern] = {"strength": strength, "count": count}

        return stats


def serialize_predictor(predictor):
    """Serialize the predictor object to store in session."""
    return base64.b64encode(pickle.dumps(predictor)).decode('utf-8')


def deserialize_predictor(serialized_predictor):
    """Deserialize the predictor object from session."""
    try:
        return pickle.loads(base64.b64decode(serialized_predictor.encode('utf-8')))
    except AttributeError:
        # If there's an error due to class changes, create a new predictor
        print("Creating new predictor due to class mismatch")
        return EnhancedPredictor()


@app.route('/')
def index():
    """Render the game page."""
    # Initialize a new predictor if none exists
    if 'predictor' not in session:
        predictor = EnhancedPredictor()
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

        # Get the predictor from session
        predictor = deserialize_predictor(session['predictor'])

        # AI makes its move based on prediction
        ai_move, confidence, strategy = predictor.get_counter_move()

        # Determine the winner
        if user_move == ai_move:
            result = 'tie'
        elif BEATS[user_move] == ai_move:
            result = 'win'
        elif BEATS[ai_move] == user_move:
            result = 'lose'
        else:
            result = 'tie'  # Fallback, should never happen

        # Update scores
        scores = session['scores']
        if result == 'win':
            scores['user'] += 1
        elif result == 'lose':
            scores['ai'] += 1
        else:
            scores['tie'] += 1

        # Update predictor with user's move and result
        predictor.update(user_move, result)

        # Store round information
        rounds = session['rounds']
        rounds.append({
            'user_move': user_move,
            'ai_move': ai_move,
            'result': result,
            'confidence': round(confidence * 100, 1),
            'strategy': strategy
        })
        if len(rounds) > 10:  # Keep only last 10 rounds
            rounds.pop(0)

        # Update session
        session['predictor'] = serialize_predictor(predictor)
        session['scores'] = scores
        session['rounds'] = rounds

        # Get additional strategy stats
        try:
            strategy_stats = predictor.get_strategy_stats()
        except AttributeError:
            # Fallback for backward compatibility
            strategy_stats = {
                "confidence": confidence,
                "strategy": strategy,
                "patterns": {}
            }

        # Return game data
        return jsonify({
            'user_move': user_move,
            'ai_move': ai_move,
            'result': result,
            'confidence': round(confidence * 100, 1),
            'strategy': strategy,
            'scores': scores,
            'move_frequencies': predictor.get_move_frequencies(),
            'transitions': predictor.get_transition_heatmap(),
            'strategy_stats': strategy_stats
        })
    except Exception as e:
        print(f"Error in play route: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset():
    """Reset the game."""
    # Create a new predictor
    predictor = EnhancedPredictor()
    session['predictor'] = serialize_predictor(predictor)
    session['scores'] = {'user': 0, 'ai': 0, 'tie': 0}
    session['rounds'] = []

    return jsonify({'message': 'Game reset successfully'})


if __name__ == '__main__':
    app.run(debug=True)