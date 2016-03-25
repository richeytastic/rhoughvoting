#pragma once
#ifndef HOUGHVOTING_RANDOM_FOREST_MT_H
#define HOUGHVOTING_RANDOM_FOREST_MT_H

#include "RandomForest.h"
using HoughVoting::RandomForest;
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>


namespace HoughVoting
{

class RandomForest_MT
{
public:
    typedef boost::shared_ptr<RandomForest_MT> Ptr;
    static Ptr create( RandomForest::Ptr forest);
    RandomForest_MT( RandomForest::Ptr forest);
    ~RandomForest_MT();

    // Multi-threaded version of forest->grow - if dontwait == true (default), function
    // will return immediately (does not block).
    // Client should call join() as soon as convenient to allow forest to complete.
    void grow( const TrainSet& trainSet, bool dontwait=true);
    void join();

    void cancel();  // Immediately cancel the background threads

private:
    RandomForest::Ptr _forest;
    boost::thread_group* _growThreads;
    vector<RandomTree*> _trees;
};  // end class

}   // end HoughVoting

#endif




