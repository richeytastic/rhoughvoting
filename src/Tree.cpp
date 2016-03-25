#include "Tree.h"
using HoughVoting::Tree;


Tree::Tree() : _trainingUpdater(NULL)
{
}   // end ctor


void Tree::setTrainingProgressUpdater( ProgressDelegate* trainingUpdater)
{
    _trainingUpdater = trainingUpdater;
}   // end setTrainingProgressUpdater
