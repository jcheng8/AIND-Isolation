"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import logging
import math
import numpy as np

logging.basicConfig(level=logging.INFO)

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


class UnknownSearchMethod(Exception):
    """Subclass base exception for unknown search method"""
    def __init__(self, method):
        self.unknown_method = method

    def __str__(self):
        return "Unknown search method {}".format(self.unknown_method)


def is_game_over(game):
    """ Check whether it's game over state, i.e either active player
    loses or wins

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    Returns
    -------
    Boolean 
        whether the game is over
    """
    return (game.is_winner(game.active_player) or
           game.is_loser(game.active_player))

def threat(game, player):
    """Evaluation function calculating the threat of player's legal moves to opponent.
    Threat is defined as cardinality of intersection of player's legal moves and opponent's
    legal moves

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        the numeric representation of threat
    """

    moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    shared_moves = [1 for move in opp_moves if move in moves] 
    return sum(shared_moves)

def monopoly(game, player):
    """Evaluation function  calculating the game score similar to `Monopoly` game:
    the  number of valuable estate player occupies so far. Here valuable estate 
    translates to the cells in the central quandrant (the /// area in the below board)
    _________________
    |   |   |   |   |
    -----------------
    |   |///|///|   |
    -----------------
    |   |///|///|   |
    -----------------
    |   |   |   |   |
    -----------------

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        the numeric representation of occupied valuable estate 
    """

    left, right = math.ceil(0.25 * game.width), int(0.75 * game.width)
    top, bottom = math.ceil(0.25 * game.height), int(0.75* game.height)
    symbol = game.get_player_symbol(player) 

    estate = sum([1 for i in range(left, right) for j in range(top, bottom) 
        if game.get_symbol_from_state(i, j) == symbol])
    return estate 

def centrality(game, player):
    """Evaluation function  calculating the centrality of given player's current
    legal moves. Centrality is defined as the distance from move's destination cell
    to the board center.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        the summation of legal moves' centrality 
    """
    legal_moves = game.get_legal_moves(player)
    center = (game.height // 2, game.width // 2)
    distance = sum([ (center[0] - move[0]) ** 2 + (center[1] - move[1]) ** 2for move in legal_moves])
    return distance
     
def entropy(game, player):
    """Evaluation function calculating the entropy of given player's legal moves.
    entropy is defined as -P(at_left) * log(P(at_left)) - P(at_right) * log(P(at_right))
    where P(at_left) is the probability of moves at the left half of board, 
    and P(at_right) is the probability of moves at the right half of board.
    P(at_left) + P(at_right) = 1.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The entropy of the specified player's legal moves.
    """
    legal_moves = game.get_legal_moves(player)
    total_moves_count = len(legal_moves)
    if total_moves_count == 0:
        return 0.0

    reference_point = (game.width // 2)
    
    left_count = sum([1 for move in legal_moves if move[1] < reference_point])
    right_count = total_moves_count - left_count

    ent = 0.0
    probs = [left_count/total_moves_count, right_count/total_moves_count]
    for p in probs:
        if p > 0:
            ent = ent + (-p * np.log(p))
    return ent
        
def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    opponent = game.get_opponent(player)
    diff_of_moves = len(game.get_legal_moves(player)) - len(game.get_legal_moves(opponent))
    #diff_of_monopoly = monopoly(game, player) - monopoly(game, game.get_opponent(player))
    threat_score = threat(game, player)
    #diff_entropy = entropy(game, game.active_player) - entropy(game, game.inactive_player)
    #return float(diff_of_moves * 0.0 + diff_of_centrality * 1)
    #return float(diff_of_moves) * 0.5 +  diff_of_centrality * 0.5
    #diff_of_centrality = centrality(game, game.inactive_player) - centrality(game, game.active_player)
    return float(diff_of_moves) * 0.5 +  threat_score * 0.5


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        if not legal_moves:
            return (-1, -1)

        # helper method to search till depth for given method
        # raise UnknownSearchMethod exception is method is unrecognized
        def search(game, method, depth):
            score = 0
            move =None

            if method == 'minimax':
                score, move = self.minimax(game, depth)
            elif method == 'alphabeta':
                score, move = self.alphabeta(game, depth)   
            else:
                raise UnknownSearchMethod(self.method)
            return score, move

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            best_move = None

            if self.iterative:
                max_depth = game.width * game.height - 2
                for d in range(0, max_depth):
                    depth = d + 1
                    _, best_move = search(game, self.method, depth)         

                    #logging.info("Iterative: time_left={} after depth={}".format(
                    #    self.time_left(), depth))

                    if self.time_left() <= 0:
                        return best_move
            else:
                _, best_move = search(game, self.method, self.search_depth)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            logging.debug("timeout when get_move. Iterative={}, method={}".format(
                self.iterative, self.method))

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.
        
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth <= 0 or is_game_over(game):
            return self.score(game, self), None

        best_move = None
        best_score = None
        if maximizing_player:
            best_score = float("-inf")
        else:
            best_score = float("inf")

        moves = game.get_legal_moves()
        for move in moves:
            game_copy = game.forecast_move(move)
            current_score, _ = self.minimax(game_copy, depth - 1, not maximizing_player)
            if maximizing_player:
                if current_score > best_score:
                    best_score = current_score
                    best_move = move
            else:
                if current_score < best_score:
                    best_score = current_score
                    best_move = move
        return best_score, best_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth <= 0 or is_game_over(game):
            return self.score(game, self), None

        best_move = None

        moves = game.get_legal_moves()

        if maximizing_player:
            for move in moves:
                game_copy = game.forecast_move(move)
                current_score, _ = self.alphabeta(game_copy, depth - 1, alpha, beta, not maximizing_player)
                if current_score > alpha:
                    alpha = current_score
                    best_move = move
                if alpha >= beta:
                    return alpha, best_move

            return alpha, best_move
        else:
            for move in moves:
                game_copy = game.forecast_move(move)
                current_score, _ = self.alphabeta(game_copy, depth - 1, alpha, beta, not maximizing_player)
                if current_score < beta:
                    beta = current_score
                    best_move = move
                if beta <= alpha:
                    return beta, best_move

            return beta, best_move

    def negamax(self, game, depth, color=1):
        """Implement the negamax search algorithm as described in wikipedia.
        
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        color : int (0 or 1)
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (1) or a minimizing layer (0)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """

        if depth <= 0 or is_game_over(game):
            return color * self.score(game, self), None

        best_score = float("-inf")
        best_move = None

        moves = game.get_legal_moves()
        for move in moves:
            game_copy = game.forecast_move(move)
            current_score, _ = -1 * self.negamax(game, depth - 1, -color)
            if current_score > best_score:
                best_score = current_score
                best_move = move
        return best_score, best_move

    def ab_negamax(self, game, depth, alpha=float("-inf"), beta=float("inf"), color=1):
        """Implement negamax search with alpha-beta pruning as described in the
        wikipedia.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        color : int (0 or 1)
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (1) or a minimizing layer (0)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """

        if depth <= 0 or is_game_over(game):
            return color* self.score(game, self), None

        best_score = float("-inf")
        best_move = None

        moves = game.get_legal_moves()
        for move in moves:
            game_copy = game.forecast_move(move)
            current_score, _ = -1 * self.ab_negamax(game_copy, depth - 1, -beta, -alpha, -color)
            if current_score > best_score:
                best_score = current_score
                best_move = move
            alpha = np.max(alpha, current_score)
            if alpha >= beta:
                return best_score, best_move

        return best_score, best_move
