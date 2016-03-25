#include "RandomForest.h"
using HoughVoting::RandomForest;
using HoughVoting::TreeOffsets;
#include <cmath>
#include <cstdlib>
#include <algorithm>


// Uniformly sample v with replacement
void sampleWithReplacement( const vector<Patch> &v, vector<Patch> &x)
{
    const int maxIdx = v.size() - 1;
    for ( int i = 0; i <= maxIdx; ++i)
        x.push_back( v[(int)((double)maxIdx * ((double)random() / RAND_MAX) + 0.5)]);
}   // end sampleWithReplacement



typedef vector<RandomTree*> ThreadLocalTrees;

void createTreesThreadFunction( int numTrees, const vector<Patch> *xs, ThreadLocalTrees *trees)
{
    for ( int i = 0; i < numTrees; ++i)
    {
        vector<Patch> boostedSet;
        sampleWithReplacement( *xs, boostedSet);
        assert( !boostedSet.empty());
        RandomTree *tree = new RandomTree( boostedSet);
        assert( tree != NULL);
        trees->push_back( tree);
    }   // end for
}   // end createTreesThreadFunction



RandomForest::RandomForest( int numTrees, const vector<Patch> &xs)
{
    boost::thread_group treeThreads;
    const int numThreads = std::min( numTrees, (int)boost::thread::hardware_concurrency());
    const int treesPerThread = numTrees / numThreads;   // integer division
    int remTrees = numTrees % numThreads;

    vector<ThreadLocalTrees*> allThreadTrees;
    for ( int i = 0; i < numThreads; ++i)
    {
        ThreadLocalTrees *threadLocalTrees = new ThreadLocalTrees;
        allThreadTrees.push_back(threadLocalTrees);
        const int nt = remTrees-- > 0 ? treesPerThread + 1 : treesPerThread;
        treeThreads.create_thread( boost::bind( &createTreesThreadFunction, nt, &xs, threadLocalTrees));
    }   // end for

    treeThreads.join_all(); // Deletes all threads

    // Add all the trees from the thread local trees to the overall list
    BOOST_FOREACH ( const ThreadLocalTrees *ts, allThreadTrees)
    {
        trees_.insert( trees_.end(), ts->begin(), ts->end());
        delete ts;
    }   // end for
}   // end ctor



RandomForest::~RandomForest()
{
    BOOST_FOREACH ( const RandomTree *tree, trees_)
        delete tree;
}   // end dtor



void calcTreeProbsThreadFunction( const cv::Mat_<float> *m, int clabel,
            const vector<const RandomTree*> *trees, int i, int sz, vector<TreeOffsets*> *offsets)
{
    const int endTree = i + sz;
    for ( ; i < endTree; ++i)
    {
        const RandomTree *tree = (*trees)[i];
        TreeOffsets *to = new TreeOffsets;
        to->offsets = NULL;
        to->prob = 0;
        to->prob = tree->lookup( *m, clabel, to->offsets);

        if ( to->offsets != NULL)
        {
            to->prob /= to->offsets->size();
            offsets->push_back(to);
        }   // end if
    }   // end foreach
}   // end calcTreeProbsThreadFunction



void RandomForest::lookup( const cv::Mat_<float> &m, int clabel, unordered_set<TreeOffsets*> &offsets) const
{
    boost::thread_group treeThreads;
    const int numTrees = trees_.size();

    const int numThreads = std::min( numTrees, (int)boost::thread::hardware_concurrency());
    const int treesPerThread = numTrees / numThreads;   // integer division
    int remTrees = numTrees % numThreads;

    vector<TreeOffsets*> *treeProbs[numThreads];    // Thread local storage

    int j = 0;  // index into trees_ for each thread
    for ( int i = 0; i < numThreads; ++i)
    {
        treeProbs[i] = new vector<TreeOffsets*>;
        const int nt = remTrees-- > 0 ? treesPerThread + 1 : treesPerThread;
        treeThreads.create_thread( boost::bind(
                    &calcTreeProbsThreadFunction, &m, clabel, &trees_, j, nt, treeProbs[i]));
        j += nt;    // index of start of next tree
    }   // end for

    treeThreads.join_all();
    for ( int i = 0; i < numThreads; ++i)
    {
        BOOST_FOREACH ( TreeOffsets *to, *treeProbs[i])
            offsets.insert( to);
        delete treeProbs[i];
    }   // end for
}   // end lookup
