% ----------------Helper Functions ----------------------------
mult(A,B,C) :- C is A * B. % Multiplies two values and stores them in C
add(A,B,C) :- C is A + B. % adds two values and stores them in C

head([Head | _], Head).   % Gets the head of a list, probably not needed

storeAsList(A, [A]).   % Converts argument A to a list
store(A,A).		% Used as a replacement for assingment

concatToList(A, B, [A | B]). % Concatenates two items into a list

zero(V, 0.0) :- V > 0.0, !.
zero(V, -0.0) :- V < 0.0, !.

% ----------------End helper functions ------------------------

hop11Activation(Net, Alpha, _,  1) :- Net > Alpha,!.
hop11Activation(Net, Alpha, Oldo,  Oldo) :- Net == Alpha,!.
hop11Activation(Net, Alpha, _, -1) :- Net < Alpha,!.

% ------------------------------------------------------------------------------

% (** Returns net activation (scalar) for a single unit using our
% list-based input and weight representation and Eqn (1) *)
netUnit([], [], 0) :- !.
netUnit([Hi | Ti], [Hw | Tw], Net) :-
  mult(Hi,Hw,X),
  netUnit(Ti,Tw,Z),
  add(X,Z,Net).

% ------------------------------------------------------------------------------

% (* Returns net activation computation for entire network
% as a vector (list) of individual unit activations *)
netAll(_,[],[]) :- !.
netAll(State,[Hw | Tw],[NetH | NetT]) :-
  netUnit(State,Hw,NetH),
  netAll(State,Tw,NetT).

% ------------------------------------------------------------------------------
% Calculate the weights
weightHelper(_,[],[]) :- !.
weightHelper( In , [ HS | TS ], [ Hw | Rw ]):-
	Hw is In * HS,
	weightHelper(In, TS, Rw),
	!.

% Base case to terminate recursion
hopTrainAHelper(H, [HAs | []], WforState):-
	zero(HAs, Z),
	storeAsList(Z,X) ,
	append(H, X, A) ,
	weightHelper(HAs, A, Weight),
	storeAsList(Weight, WforState),!.

% Generate the weights for a single state recursively
hopTrainAHelper(H, [HAs | TAs], WforState):-
	zero(HAs, Z),
	concatToList(Z,TAs,X),
	append(H , X , Start) ,
	append(H ,[HAs], Heads),
	weightHelper(HAs , Start , Weight),
	hopTrainAHelper(Heads , TAs , Weight2),
	concatToList(Weight, Weight2, WforState).

% (** Returns weight matrix for only one stored state,
% used as a ’warmup’ for the next function *)
hopTrainAstate(AState , WforState) :-
	hopTrainAHelper(_ , AState , WforState),!.

% ------------------------------------------------------------------------------

% Takes in two single lists and sums each element by element and returns a single list
reduce([Head1 | []], [Head2 | []], Result):-
	add(Head1,Head2,R1),
	storeAsList(R1, Result),
	!.

% Takes in two single lists and sums each element by element and returns a single list
reduce([Head1 | Tail1], [Head2 | Tail2], Result):-
	add(Head1 , Head2 , R1),
	reduce(Tail1, Tail2, R2),
	concatToList(R1, R2, Result),
	!.

% Takes in two list of lists and reduces to one list of lists
reduceList([Head1 | []],[Head2 | []], Result):-
	reduce(Head1, Head2, R1),
	storeAsList(R1, Result).

% Takes in two list of lists and reduces to one list of lists
reduceList([Head1 | Tail1], [Head2 | Tail2], Result):-
	reduce(Head1, Head2, R1),
	reduceList(Tail1, Tail2, R2),
	concatToList(R1, R2, Result).

% (** This returns weight matrix, given a list of stored states
% (shown previously) using Eqns (4) and (5) *)
hopTrain([HeadOfStates | []], WeightMatrix ):-
	hopTrainAstate(HeadOfStates, WeightMatrix),
	!.

% (** This returns weight matrix, given a list of stored states
% (shown previously) using Eqns (4) and (5) *)
hopTrain([HeadOfStates | TailOfStates], WeightMatrix ):-
	hopTrainAstate(HeadOfStates, R1),
	hopTrain(TailOfStates, R2),
	reduceList(R1, R2, WeightMatrix),
	!.

hopTrain([],_).
% ------------------------------------------------------------------------------
% (** Returns next state vector  *) base case to terminate recursion
nextState([Hc | Tc],[Hw | []], Alpha, Next) :-
  netUnit([Hc | Tc], Hw, A),
  hop11Activation(A, Alpha, Hc, B),
  storeAsList(B,Next),
  !.

% (** Returns next state vector  *)
nextState([Hc | Tc],[Hw | Tw], Alpha, Next) :-
  netUnit([Hc | Tc], Hw, A),
  hop11Activation(A, Alpha, Hc, B),
  nextState([Hc | Tc], Tw, Alpha, C),
  append([B], C, Next).
% ------------------------------------------------------------------------------

% (** Returns network state after n time steps; i.e.,  update of N time steps *)
updateN(CurrentState, WeightMatrix, Alpha, N, ResultState) :-
	N > 0,
	nextState(CurrentState, WeightMatrix, Alpha, Next),
	I is N - 1,
	updateN(Next, WeightMatrix, Alpha, I, ResultState), !;
	store(CurrentState, ResultState).
