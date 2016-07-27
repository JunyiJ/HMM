# Implementation of Forward-Backward algorithm, Viterbi algorithm in Hidden Markov model using Python

##Introduction
*Hidden Markov Model (HMM)* is a statistical Markov model with unobserved states.
It has been widely used in different fields to solve real life problems, such as 
speech recognision and handwriting. This project aims to implement several basic 
algorithm widely used in HMM using python (numpy), including the *forward-backward
 algorithm* and *Viterbi algorithm*

## Usage
Here is the implemention of the forward-backward algorithm and Viterbi algorithm to 
solve two commen Hidden Markov Model(HMM) problems with python (ipython):
1)Find the model lamda=(A,B,p) given an observation sequence O and dimensions N and M.
2)Find the hidden state sequence given an observation sequence O and model lamda=(A,B,p)

A: Transition ;
B: Emission matrix;
p: Initial prob. distribution;

##Reference
This implementation is based on Mark Stamp's A Revealing Introduction to Hidden Markov Models 
(Department of Computer Science, San Jose State University)
