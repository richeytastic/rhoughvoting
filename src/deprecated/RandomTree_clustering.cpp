/************************************************************************
 * Copyright (C) 2017 Richard Palmer
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ************************************************************************/

#include "RandomTree.h"
using HoughVoting::RandomTree;
using HoughVoting::Patch;
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
using std::cerr;
using std::endl;
#include <boost/unordered_map.hpp>
using boost::unordered_map;
#include <map>  // For std::pair



class OffsetGroup
{
private:
    const int numAngleBins_;
    const double radsPerBin_;
    int numExs_;
    double *offsetDistances_;
    double *weights_;
    vector<cv::Vec2d> offsets_;

public:
    OffsetGroup( int numAngleBins)
        : numAngleBins_(numAngleBins), radsPerBin_(360./numAngleBins * M_PI/180), numExs_(0),
          offsetDistances_(NULL), weights_(NULL)
    {
        offsetDistances_ = (double*)calloc( numAngleBins, sizeof(double));
        weights_ = (double*)calloc( numAngleBins, sizeof(double));
    }   // end ctor

    ~OffsetGroup()
    {
        free( offsetDistances_);
        free( weights_);
    }   // end dtor

    void add( double simMeasure/*between 0 and 1*/, const cv::Vec2d &offset)
    {
        numExs_++;
        offsets_.push_back(offset);
        return;

        const double rads = atan2( offset[1], offset[0]) + M_PI;    // Result in range [0,2*M_PI]
        const int binIdx = (int)(rads/radsPerBin_ + 0.5) % numAngleBins_;
        const double dist = sqrt(pow(offset[0],2) + pow(offset[1],2));
        offsetDistances_[binIdx] += simMeasure * dist;
        weights_[binIdx] += simMeasure;
    }   // end add

    void calcOffsets()
    {
        return;

        offsets_.clear();
        for ( int i = 0; i < numAngleBins_; ++i)
        {
            int j = i - 1;
            if ( j < 0)
                j = numAngleBins_ - 1;

            int k = i + 1;
            if ( k == numAngleBins_)
                k = 0;

            const double d = (offsetDistances_[j]/2 + offsetDistances_[i] + offsetDistances_[k]/2)
                           / (weights_[j]/2 + weights_[i] + weights_[k]/2);

            const double rads = i*radsPerBin_;
            const cv::Vec2d vec( d*cos(rads), d*sin(rads));
            offsets_.push_back( vec);
        }   // end for
    }   // end calcOffsets

    const vector<cv::Vec2d>* const getOffsets() const
    {
        assert( !offsets_.empty());
        return &offsets_;
    }   // end getOffsets

    int getCount() const
    {
        return numExs_;
    }   // end getCount
};  // end class



typedef std::pair<int, OffsetGroup*> OGPair;


class RandomTree::Node
{
private:
    int level_; // Node level
    cv::Mat_<float> cluster_;
    Node *leftChild_;
    Node *rightChild_;

    int totCount_;  // Total number of objects in node (sum of elements of objCounts_)
    vector<int> *objCounts_; // Indexed by class ID
    unordered_map<int, OffsetGroup*> *offsets_;     // Offsets per object class
    unordered_map<int, cv::Mat_<float> > icovs_;    // Inverted covariance matrices per object class


    Node( int lev, const cv::Mat_<float> &cluster, const vector<Patch> &xs)
        : level_(lev), cluster_(cluster), leftChild_(NULL), rightChild_(NULL),
          totCount_(0), objCounts_(NULL), offsets_(NULL)
    {
        const int numExs = xs.size();
        const int fvSz = xs[0].descriptor.cols;

        // Is this a leaf node?
        if ( level_ == MAX_LEVEL || numExs <= MAX_PATCHES_PER_NODE)
        {
            totCount_ = numExs;
            objCounts_ = new vector<int>;
            offsets_ = new unordered_map<int, OffsetGroup*>;
            unordered_map<int, cv::Mat_<float> > dgrps;  // Groups of descriptors per class

            BOOST_FOREACH ( const Patch &p, xs)
            {
                const int clab = p.clabel;

                // Increment the object count for this object class
                while ( objCounts_->size() <= clab)
                    objCounts_->push_back(0);
                (*objCounts_)[clab]++;

                // Append the descriptor from this patch to the group of descriptors
                // for the corresponding class.
                if ( dgrps.count(clab) == 0)
                    dgrps[clab] = cv::Mat_<float>(0, fvSz);
                dgrps[clab].push_back(p.descriptor);

                if ( clab == 0) // Offsets not collected for non-objects
                    continue;

                // Create a new offset group if it doesn't already exist
                if ( offsets_->count(clab) == 0)
                {
                    const int numAngles = 9 + (int)(9. * (double)random() / RAND_MAX + 0.5);
                    (*offsets_)[clab] = new OffsetGroup( numAngles);    // Random number of angles between 9 and 18
                }   // end if
            }   // end foreach

            // Calculate the inverted covariance matrices
            const int flags = CV_COVAR_NORMAL | CV_COVAR_USE_AVG | CV_COVAR_SCALE | CV_COVAR_ROWS;
            typedef std::pair<int, cv::Mat_<float> > DPair;
            BOOST_FOREACH ( const DPair &dp, dgrps)
            {
                const int clab = dp.first;
                cv::Mat_<float> cov( fvSz, fvSz);
                cv::calcCovarMatrix( dp.second, cov, cluster_, flags, CV_32F);
                icovs_[clab] = cv::Mat_<float>(fvSz, fvSz);
                cv::invert( cov, icovs_[clab], cv::DECOMP_SVD);
            }   // end foreach

            // Add the patch offsets to the corresponding offset groups using Mahalanobis distance as weight
            BOOST_FOREACH ( const Patch &p, xs)
            {
                const int clab = p.clabel;
                if ( clab == 0)
                    continue;

                // Collect offset into corresponding object class offset group
                const cv::Mat_<float> d = p.descriptor - cluster_;
                const double val = cv::Mahalanobis( d, d, icovs_.at(clab));
                (*offsets_)[clab]->add( 1./(val+1), p.offset);
            }   // end foreach

            // Finalise offset calculations
            BOOST_FOREACH ( const OGPair &og, *offsets_)
                og.second->calcOffsets();
        }   // end else
        else
            doChildrenClustering( xs);
    }   // end ctor


    void doChildrenClustering( const vector<Patch> &xs)
    {
        const int numExs = xs.size();
        const cv::Mat &fv = xs[0].descriptor;
        assert( fv.type() == CV_32FC1);
        assert( fv.rows == 1);

        // Copy descriptors to single matrix for k-means
        cv::Mat_<float> samples( numExs, fv.cols);
        for ( int i = 0; i < numExs; ++i)
            xs[i].descriptor.copyTo( samples.row(i));

        const int k = 2;
        cv::Mat_<int> labels( numExs, 1);
        cv::Mat_<float> clusters( k, fv.cols);    // Cluster centres
        const int flags = cv::KMEANS_PP_CENTERS;
        const int nattempts = 10;
        const cv::TermCriteria tc( cv::TermCriteria::COUNT, nattempts, 0);
        cv::kmeans( samples, k, labels, tc, nattempts, flags, clusters);

        vector<Patch> leftPatches, rightPatches;
        for ( int i = 0; i < numExs; ++i)
        {
            if ( labels.at<int>(i,0) == 0)
                leftPatches.push_back( xs[i]);
            else
                rightPatches.push_back( xs[i]);
        }   // end for

        const int nextLevel = level_ + 1;

        if ( !leftPatches.empty())
            leftChild_ = new Node( nextLevel, clusters.row(0), leftPatches);

        if ( !rightPatches.empty())
            rightChild_ = new Node( nextLevel, clusters.row(1), rightPatches);
    }   // end doChildrenClustering


public:
    static const int MAX_LEVEL;
    static const int MAX_PATCHES_PER_NODE;

    Node( const vector<Patch> &xs)
        : level_(0), leftChild_(NULL), rightChild_(NULL),
          totCount_(0), objCounts_(NULL), offsets_(NULL)
    {
        doChildrenClustering( xs);
    }   // end ctor


    ~Node()
    {
        if ( leftChild_ != NULL)
            delete leftChild_;
        if ( rightChild_ != NULL)
            delete rightChild_;
        if ( objCounts_ != NULL)
            delete objCounts_;

        if ( offsets_ != NULL)
        {
            BOOST_FOREACH ( const OGPair &og, *offsets_)
                delete og.second;
            delete offsets_;
        }   // end if
    }   // end dtor


    int calcItemBalance() const
    {
        if ( isLeaf())
            return totCount_;

        int cnt = 0;
        if ( leftChild_ != NULL)
            cnt -= leftChild_->calcItemBalance();
        if ( rightChild_ != NULL)
            cnt += rightChild_->calcItemBalance();
        return cnt;
    }   // end calcItemBalance


    int calcNodeBalance() const
    {
        if ( isLeaf())
            return 1;

        int nb = 0;
        if ( leftChild_ != NULL)
            nb -= leftChild_->calcNodeBalance();
        if ( rightChild_ != NULL)
            nb += rightChild_->calcNodeBalance();
        return nb;
    }   // end calcNodeBalance


    int countItems() const
    {
        if ( isLeaf())
            return totCount_;

        int cnt = 0;
        if ( leftChild_ != NULL)
            cnt += leftChild_->countItems();
        if ( rightChild_ != NULL)
            cnt += rightChild_->countItems();
        return cnt;
    }   // end countItems


    int countNodes() const
    {
        int cnt = 1;
        if ( leftChild_ != NULL)
            cnt += leftChild_->countNodes();
        if ( rightChild_ != NULL)
            cnt += rightChild_->countNodes();
        return cnt;
    }   // end countChildNodes


    int countObjects() const
    {
        if ( isLeaf())
            return totCount_ - (*objCounts_)[0];

        int cnt = 0;
        if ( leftChild_ != NULL)
            cnt += leftChild_->countObjects();
        if ( rightChild_ != NULL)
            cnt += rightChild_->countObjects();
        return cnt;
    }   // end countObjects


    int calcHeight() const
    {
        int llev = level_;
        if ( leftChild_ != NULL)
            llev = leftChild_->calcHeight();
        int rlev = llev;
        if ( rightChild_ != NULL)
            rlev = rightChild_->calcHeight();
        return std::max( llev, rlev);
    }   // end calcHeight


    int getLeafCount() const
    {
        int leaves = 0;
        if ( isLeaf())
            leaves = 1;
        else
        {
            if ( leftChild_ != NULL)
                leaves += leftChild_->getLeafCount();
            if ( rightChild_ != NULL)
                leaves += rightChild_->getLeafCount();
        }   // end else
        return leaves;
    }   // end getLeafCount


    bool isLeaf() const
    {
        return leftChild_ == NULL && rightChild_ == NULL;
    }   // end isLeaf


    double findLeaf( const cv::Mat_<float> &fv, int clab, const vector<cv::Vec2d>* &osets) const
    {
        double objProp = 0;

        if ( isLeaf())
        {
            if ( offsets_->count(clab) == 1)
            {
                osets = (*offsets_)[clab]->getOffsets();
                const cv::Mat_<float> d = fv - cluster_;
                const double mhlb = cv::Mahalanobis( d, d, icovs_.at(clab));
                objProp = (double)(*objCounts_)[clab] / totCount_ * 1./(mhlb+1);
            }   // end if
        }   // end if
        else
        {
            // If both child nodes exist we must test which path to descend
            if ( leftChild_ != NULL && rightChild_ != NULL)
            {
                const double leftSSD = cv::norm( fv - leftChild_->cluster_, cv::NORM_L2);
                const double rightSSD = cv::norm( fv - rightChild_->cluster_, cv::NORM_L2);

                if ( leftSSD < rightSSD)
                    objProp = leftChild_->findLeaf( fv, clab, osets);
                else
                    objProp = rightChild_->findLeaf( fv, clab, osets);
            }   // end if
            else if ( leftChild_ != NULL)   // Otherwise the choice is simply left...
                objProp = leftChild_->findLeaf( fv, clab, osets);
            else if ( rightChild_ != NULL)  // or right!
                objProp = rightChild_->findLeaf( fv, clab, osets);
        }   // end else

        assert( objProp >= 0 && objProp <= 1);
        return objProp;
    }   // end findLeaf


    void calcLeafDistribution( vector<int> &levLeaves) const
    {
        if ( isLeaf())
        {
            while ( levLeaves.size() <= level_)
                levLeaves.push_back(0);
            levLeaves[level_]++;
        }   // end if
        else
        {
            if ( leftChild_ != NULL)
                leftChild_->calcLeafDistribution( levLeaves);
            if ( rightChild_ != NULL)
                rightChild_->calcLeafDistribution( levLeaves);
        }   // end else
    }   // end calcLeafDistribution


    void calcObjectDistribution( vector<int> &levObjs) const
    {
        if ( isLeaf())
        {
            while ( levObjs.size() <= level_)
                levObjs.push_back(0);
            levObjs[level_] += totCount_ - (*objCounts_)[0];
        }   // end if
        else
        {
            if ( leftChild_ != NULL)
                leftChild_->calcObjectDistribution( levObjs);
            if ( rightChild_ != NULL)
                rightChild_->calcObjectDistribution( levObjs);
        }   // end else
    }   // end calcObjectDistribution
};  // end class



const int RandomTree::Node::MAX_LEVEL = 30;
const int RandomTree::Node::MAX_PATCHES_PER_NODE = 20;


RandomTree::RandomTree( const vector<Patch> &xs)
    : root_(NULL)
{
    assert( !xs.empty());
    root_ = new RandomTree::Node( xs);
}   // end ctor



RandomTree::~RandomTree()
{
    if ( root_ != NULL)
        delete root_;
}   // end dtor



double RandomTree::lookup( const cv::Mat_<float> &fv, int clab, const vector<cv::Vec2d>* &offsets) const
{
    return root_->findLeaf( fv, clab, offsets);
}   // end lookup



int RandomTree::calcNodeBalance() const
{
    return root_->calcNodeBalance();
}   // end calcNodeBalance



int RandomTree::calcItemBalance() const
{
    return root_->calcItemBalance();
}   // end calcItemBalance



int RandomTree::countNodes() const
{
    return root_->countNodes();
}   // end countNodes



int RandomTree::countItems() const
{
    return root_->countItems();
}   // end countItems



int RandomTree::countObjects() const
{
    return root_->countObjects();
}   // end countObjects



int RandomTree::calcHeight() const
{
    return root_->calcHeight() + 1;
}   // end calcHeight



int RandomTree::calcLeaves() const
{
    return root_->getLeafCount();
}   // end calcLeaves



int RandomTree::calcLeafDistribution( vector<int> &levLeaves) const
{
    root_->calcLeafDistribution( levLeaves);
    return levLeaves.size();
}   // end calcLeafDistribution



int RandomTree::calcObjectDistribution( vector<int> &levObjs) const
{
    root_->calcObjectDistribution( levObjs);
    return levObjs.size();
}   // end calcObjectDistribution
