#include "SupervisedCodebook.h"
using HoughVoting::SupervisedCodebook;
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;


// Calculate the mean feature vector and offset from the given feature extracts (assumes one offset per extract)
cv::Vec2f SupervisedCodebook::calcMeans( int clusterId, const vector<const FeatureExtract*>& v, cv::Mat_<float>& meanFV)
{
    const int numXs = v.size();
    assert( numXs >= 1);
    meanFV = cv::Mat_<float>::zeros( v[0]->fv.size());
    cv::Vec2f meanOffset = v[0]->objVote->offset[0];
    moffsets_[clusterId].push_back(meanOffset);

    for ( int i = 1; i < numXs; ++i)
    {
        meanFV += v[i]->fv;
        meanOffset += v[i]->objVote->offset[0];
        moffsets_[clusterId].push_back(v[i]->objVote->offset[0]);
    }   // end for

    meanFV /= numXs;
    meanOffset /= numXs;
    return meanOffset;
}   // end calcMeans



int calcQuadrant( const cv::Vec2f& v)
{
    int quad = 0;

    // Offset points FROM patch TO centre, so determine quadrant based on sign of dims
    if ( v[0] > 0) // To the left of centre
        quad = v[1] < 0 ? 1 : 2;
    else    // To the right of centre (or on centre line)
        quad = v[1] < 0 ? 0 : 3;

    return quad;
}   // end calcQuadrant



int calcSection( const cv::Vec2f& v, int quad)
{
    const bool innerCol = fabs(v[0]) < 0.25;
    const bool innerRow = fabs(v[1]) < 0.25;

    int sec = 0;
    switch ( quad)
    {
        case 0:
            sec = innerCol ? 1 : 0;
            if ( innerRow) sec = innerCol ? 2 : 3;
            break;
        case 1:
            sec = innerCol ? 0 : 1;
            if ( innerRow) sec = innerCol ? 3 : 2;
            break;
        case 2:
            sec = innerCol ? 3 : 2;
            if ( innerRow) sec = innerCol ? 0 : 1;
            break;
        case 3:
            sec = innerCol ? 2 : 3;
            if ( innerRow) sec = innerCol ? 1 : 0;
            break;
        default:
            assert(false);
    }   // end switch

    return sec;
}   // end calcSection



SupervisedCodebook::SupervisedCodebook( const unordered_set<FeatureExtract*>* xs)
{
    // |1|0|
    // |2|3|
    unordered_map<int, unordered_map<int, vector<const FeatureExtract*> > > fgSecs;
    for ( int i = 0; i < 4; ++i)
    {
        fgSecs[i] = unordered_map<int, vector<const FeatureExtract*> >();
        for ( int j = 0; j < 4; ++j)
            fgSecs[i][j] = vector<const FeatureExtract*>();
    }   // end for

    vector<int> quadCnt(4);
    // Separate out foreground from background extracts and partition the fg extracts
    // according to their offset vectors to the object centre. I.e. fg extracts are
    // clustered according to the position relative to the object they are voting for.
    vector<const FeatureExtract*> bgs;  // Background examples (class 0)
    BOOST_FOREACH ( const FeatureExtract* x, *xs)
    {
        if ( x->objVote->classId == 0)
            bgs.push_back(x);
        else
        {
            // Only a single offset (the centroid) allowed for now
            const int quad = calcQuadrant( x->objVote->offset[0]);
            const int section = calcSection( x->objVote->offset[0], quad);
            fgSecs[quad][section].push_back(x);
            quadCnt[quad]++;
        }   // end else
    }   // end foreach

    for ( int i = 0; i < 4; ++i)
        cout << "Quad " << i << " size = " << quadCnt[i] << endl;

    const int fgCount = xs->size() - bgs.size();    // Total number of foreground extracts
    pF_ = (float)(fgCount + 1) / (xs->size() + 2); // P(F=1)

    vector<int> cCounts;    // Cluster extract counts (fg and bg extracts)

    // Calculate the mean feature vector and offset for each section of each quadrant
    moffsets_.resize(16);
    int clusterId = 0;
    for ( int i = 0; i < 4; ++i)
    {
        for ( int j = 0; j < 4; ++j)
        {
            cv::Mat_<float> meanFV;
            cv::Vec2f meanOffset(-2,-2);    // Default denotes out of range

            moffsets_[clusterId] = vector<cv::Vec2f>();
            const int fgSz = fgSecs[i][j].size(); // Number of fg extracts in this cluster
            if ( fgSz > 0)
                meanOffset = calcMeans( clusterId, fgSecs[i][j], meanFV);
            assert( moffsets_[clusterId].size() == fgSz);
            meanFVs_.push_back( meanFV);
            offsets_.push_back( meanOffset);

            pQi_F_.push_back( float(fgSz + 1) / (fgCount + 16));    // P(Qi|F=1)
            cCounts.push_back( fgSz);    // Will be added to with nearest background examples
            clusterId++;
        }   // end for
    }   // end for

    // Add each background example randomly(ish) to a cluster
    int idx = 0;
    BOOST_FOREACH ( const FeatureExtract* x, bgs)
    {
        cCounts[idx]++;
        idx = (idx + 1) % 16;
    }   // end foreach

    cout << "P(F=1) = " << pF_ << endl;
    // Calculate P(Qi) for every cluster Qi
    for ( int i = 0; i < 16; ++i)
    {
        pQi_.push_back( float(cCounts[i] + 1) / (xs->size() + 16));
        const int nfg = moffsets_[i].size();
        const int nbg = cCounts[i] - nfg;
        cout << "Cluster " << i << ") FG/(FG+BG) =  " << nfg << "/" << (nfg+nbg)
             << "; P(Q_" << i << ") = " << pQi_[i]
             << "; P(Q_" << i << "|F=1) = " << pQi_F_[i] << endl;
    }   // end for
}   // end ctor



float SupervisedCodebook::lookupVote( int cid, const cv::Mat& fv, cv::Vec2f& offset) const
{
    int i;
    const float p = calcProb( cid, fv, i);

    offset = offsets_[i];
    return p;
}   // end lookupVote



float SupervisedCodebook::lookupVotes( int cid, const cv::Mat& fv, const vector<cv::Vec2f>** offsets) const
{
    int i;
    const float p = calcProb( cid, fv, i);

    *offsets = &moffsets_[i];
    return p;
}   // end lookupVotes



float SupervisedCodebook::lookupVotes( int cid, const cv::Mat& fv, int numOffsets, vector<cv::Vec2f>& offsets) const
{
    int i;
    const float p = calcProb( cid, fv, i);

    const int sz = moffsets_[i].size();
    for ( int j = 0; j < numOffsets; ++j)
        offsets.push_back( moffsets_[i][random() % sz]);
    return p;
}   // end lookupVotes



float SupervisedCodebook::calcProb( int cid, const cv::Mat& fv, int& idx) const
{
    assert( cid == 0 || cid == 1);
    double ssd;
    const int i = findClosestCluster( fv, &ssd);
    //float pQi_F = pF_/pQi_[i] * pQi_F_[i] * ssd; // Calc P(Qi|V=1)
    float pQi_F = pF_/pQi_[i] * pQi_F_[i]; // Calc P(Qi|V=1)

    if ( cid == 0)
        pQi_F = 1. - pQi_F;
    idx = i;
    return pQi_F;
}   // end calcProb



int SupervisedCodebook::findClosestCluster( const cv::Mat_<float>& fv, double* ssdOut) const
{
    double minSSD = DBL_MAX;
    double totSSD = 0;
    int besti = -1;
    for ( int i = 0; i < 16; ++i)
    {
        const cv::Mat fvd = fv - meanFVs_[i];
        double ssd = cv::norm( fvd, cv::NORM_L2);
        totSSD += ssd;

        if ( ssd < minSSD)
        {
            minSSD = ssd;
            besti = i;
        }   // end if
    }   // end for

    if ( ssdOut != NULL)
        *ssdOut = 1. - minSSD/totSSD;

    return besti;
}   // end findClosestCluster
