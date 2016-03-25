#include "RandomForest_MT.h"
using HoughVoting::RandomForest_MT;
#include <cassert>


RandomForest_MT::Ptr RandomForest_MT::create( RandomForest::Ptr rf)
{
    return RandomForest_MT::Ptr( new RandomForest_MT(rf));
}   // end create


RandomForest_MT::RandomForest_MT( RandomForest::Ptr rf) : _forest(rf), _growThreads(NULL)
{
    assert( _forest != NULL);
}   // end ctor


RandomForest_MT::~RandomForest_MT()
{
    join();
}   // end dtor



void RandomForest_MT::grow( const TrainSet& trainSet, bool dontwait)
{
    cancel();  // Ensure previous thread group destroyed
    const int numTrees = _forest->getNumTrees();
    _trees.resize(numTrees);
    _growThreads = new boost::thread_group;
    for ( int i = 0; i < numTrees; ++i)
    {
        _trees[i] = _forest->getTree(i);
        assert( _trees[i] != NULL);
        const int randInit = time(0) + i;
        _growThreads->create_thread( boost::bind( &RandomTree::grow, _trees[i], &trainSet, randInit));
    }   // end for

    if ( !dontwait)
        join();
}   // end grow_mt



void RandomForest_MT::join()
{
    if ( _growThreads != NULL)
    {
        _growThreads->join_all();
        delete _growThreads;
        _growThreads = NULL;
    }   // end if
}   // end join



void RandomForest_MT::cancel()
{
    if ( _growThreads != NULL)
    {
        _growThreads->interrupt_all();
        _growThreads->join_all();
        delete _growThreads;
        _growThreads = NULL;
    }   // end if
}   // end cancel
