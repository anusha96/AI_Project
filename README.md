# AI_Project
Implementation of Snake and Ladder game in AI

Authors:
Anusha Prakash
Janani Hemachandran
Madhura Anand

Affliations: Syracuse University, New York

![snakeandlad](https://user-images.githubusercontent.com/13360635/205380027-186e29aa-8141-452b-9e30-08739d8e3e55.jpg)

This is a text based implementation of the Snake and Ladder game but with a twist!

This game can be played between a human player and with an AI agent.

The Rules of the game is as follows,

    Rules:
      1. Initally both the players are at starting position i.e. 0. 
         Take it in turns to roll the dice. 
         Move forward the number of spaces shown on the dice.
      2. If you lands at the bottom of a ladder, you can move up to the top of the ladder.
      3. If you lands on the head of a snake, you must slide down to the bottom of the snake.
      4. The first player to get to the FINAL position is the winner.
      5. enter to roll the dice.
      6. Next, the player can select how many moves to move on the board once the dice is rolled, i.e if the dice rolled is 3 then 
         the player can move 0,1,2 or 3 moves on the board.
         
    The file SnakeandLadderAI.py is the version of the game that's played between two AI agents, (The experimental section)
    The two AI's here are Tree based and Neural Network based AI.
    100 different random samples are generated that represent every instance in the game and experimented against the Agents,
    the  results from 3 different Neural Network configurations like layers and learning rate are then recorded.
    
  
     The file snakeandladder_humans.py is the version of the game that's played between an AI agent and a human. In this version of the game the player gets to choose between two AI agents,
     1. Tree Based AI
     2. Neural Network Based AI
    The player can choose the board size between 75, 100 and 200. 
    The player can also choose the dice range once the dice is rolled.
  
  The snake and ladder environment can be changed.
  
  The files NN_Model.py, NNVersion2.py and NNVersion3.py are different versions of Neural Network on a mid-size problem.
