allAct([-10.0,5.0,-1.0,2.3,0.0,7.7,-3.5]).
data(oi,[-1.0,-1.0]).
data(we,[[0.0,-1.0],[-1.0,0.0]]). /* for cycle */
data(os1,[1.0, -1.0, 1.0, -1.0]).
data(os2,[-1.0, -1.0, 1.0, -1.0]).
data(os3,[-1.0, -1.0, 1.0, 1.0]).
data(os4,[1.0, 1.0, 1.0, 1.0]).
data(os5,[-1.0, -1.0, -1.0, -1.0]).
data(os6,[1.0, 1.0, -1.0, -1.0]).
data(o1o2o3,[[1.0, -1.0, 1.0, -1.0],[-1.0, -1.0, 1.0, -1.0],[-1.0, -1.0, 1.0, 1.0]]).
data(o1o1o1o2o3,[[1.0, -1.0, 1.0, -1.0],[1.0, -1.0, 1.0, -1.0],[1.0, -1.0, 1.0,
-1.0],[-1.0, -1.0, 1.0, -1.0],[-1.0, -1.0, 1.0, 1.0]]).
data(w,[[0.0, -1.0, 1.0, -1.0], [-1.0, 0.0, -1.0, 1.0], [1.0, -1.0, 0.0, -1.0],
[-1.0, 1.0, -1.0, 0.0]]).

% ----------------Helper Functions ----------------------------
mult(A,B,C) :- C is A * B.
add(A,B,C) :- C is A + B.

head([Head|_], Head).   % Gets the head of a list, probably not needed

storeAsList(A,[A]).   % Converts argument A to a list
store(A,A).

concatToList(A,B,[A|B]).

helper1(_,[],[]).
helper1( H , [ HAs | TAs ], [ Ho | Ro ]):-
	Ho is H * HAs,
	helper1( H , TAs , Ro ),
	!.

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
netUnit([Hi|Ti],[Hw|Tw],Net) :-
  mult(Hi,Hw,X),
  netUnit(Ti,Tw,Z),
  add(X,Z,Net).

% ------------------------------------------------------------------------------

% (* Returns net activation computation for entire network
% as a vector (list) of individual unit activations *)
netAll(_,[],[]) :- !.
netAll(State,[Hw|Tw],[NetH|NetT]) :-
  netUnit(State,Hw,NetH),
  netAll(State,Tw,NetT).

% ------------------------------------------------------------------------------
% Calculate the weights
weight(_,[],[]) :- !.
weight( In , [ HS | TS ], [ Hw | Rw ]):-
	Hw is In * HS,
	% mult(In,Hs,Hw),
	weight(In , TS , Rw),
	!.
	
% Base case to terminate recursion
hopTrainAHelper( H , [ HAs | []], WforState):-
	zero(HAs, Z),
	storeAsList(Z,X) ,
	append( H , X , A ) ,
	weight( HAs , A , Weight ),
	storeAsList(Weight, WforState),!.

% Generate the weights for a single state recursively
hopTrainAHelper(H, [ HAs | TAs ], WforState):-
	zero(HAs, Z),
	concatToList(Z,TAs,X),
	append( H , X , Start ) ,
	append( H ,[ HAs ], Heads ),
	weight( HAs , Start , Weight ),
	hopTrainAHelper( Heads , TAs , Weight2 ),
	concatToList(Weight, Weight2, WforState).

% (** Returns weight matrix for only one stored state,
% used as a ’warmup’ for the next function *)
hopTrainAstate( AState , WforState ) :-
	hopTrainAHelper( _ , AState , WforState ),!.

% ------------------------------------------------------------------------------
nextState([Hc|Tc],[Hw|[]], Alpha, Next) :-
  netUnit([Hc|Tc], Hw, A),
  hop11Activation(A, Alpha, Hc, B),
  storeAsList(B,Next),
  !.
nextState([Hc|Tc],[Hw|Tw],Alpha,Next) :-
  netUnit([Hc|Tc], Hw, A),
  hop11Activation(A, Alpha, Hc, B),
  nextState([Hc|Tc], Tw, Alpha, C),
  append([B], C, Next).
% ------------------------------------------------------------------------------

%
% updateN(CurrentState,WeightMatrix,Alpha,N,ResultState)
