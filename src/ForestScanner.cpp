#include "ForestScanner.h"
using HoughVoting::ForestScanner;
#include <cassert>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <iostream>
using std::cerr;
using std::endl;
#include <list>
using std::list;


ForestScanner::ForestScanner( const RandomForest::Ptr forest, const vector<FeatureExtractor::Ptr>& fxs,
                              const cv::Mat_<float>& rngMap, const cv::Mat_<byte>& mask, bool useDepthWeighting)
    : _forest(forest), _fxs(fxs), _rngMap(rngMap), _mask(mask), _imgRct(0,0,mask.cols,mask.rows),
      _useDepthWeighting(useDepthWeighting), _responseRect(0,0,rngMap.cols,rngMap.rows)
{
    assert( forest != NULL);
    // Get the minimum sampling dims over all FXs
    _minSamplingDims = cv::Size(0,0);
    for ( int i = 0; i < fxs.size(); ++i)
    {
        assert( fxs[i]->isPreProcessed());
        const cv::Size msz = fxs[i]->getMinSamplingDims();
        _minSamplingDims.width = std::max<int>( _minSamplingDims.width, msz.width);
        _minSamplingDims.height = std::max<int>( _minSamplingDims.height, msz.height);
    }   // end for
}   // end ctor



ForestScanner::~ForestScanner()
{}   // end dtor



const cv::Mat_<float> ForestScanner::scaleScan( cv::Size2f realPatchSz)
{
    _responseMap = cv::Mat_<float>::zeros( _rngMap.size());

    assert( _rngMap.rows == _mask.rows);
    assert( _rngMap.cols == _mask.cols);
    assert( _imgRct.width == 512);
    assert( _imgRct.height == 512);
    RFeatures::AdaptiveDepthPatchScanner adps( _rngMap, realPatchSz, this);
    adps.scan( _mask);

    _responseMap /= getNumTrees();
    return _responseMap;
}   // end scaleScan



const cv::Mat_<float> ForestScanner::fixedScan( cv::Size2i pxlPatchDims)
{
    _responseMap = cv::Mat_<float>::zeros( _rngMap.size());

    const cv::Mat_<byte> mask = _mask;
    const cv::Size imgSz = mask.size();
    const int halfPatchHeight = pxlPatchDims.height/2;
    const int halfPatchWidth = pxlPatchDims.width/2;
    cv::Rect fxdPatch( 0, 0, pxlPatchDims.width, pxlPatchDims.height);  // Patch size is always fixed
    cv::Point p;
    for ( int i = halfPatchHeight; i < imgSz.height - halfPatchHeight; ++i)
    {
        p.y = i;
        fxdPatch.y = i - halfPatchHeight;
        for ( int j = halfPatchWidth; j < imgSz.width - halfPatchWidth; ++j)
        {
            p.x = j;
            fxdPatch.x = j - halfPatchWidth;
            if ( mask.at<byte>(p))
                process( p, _rngMap.at<float>(p), fxdPatch);
        }   // end for - cols
    }   // end for - rows

    _responseMap /= getNumTrees();
    return _responseMap;
}   // end fixedScan



/*
// public
cv::Mat_<float> ForestScanner::createSupportMap() const
{
    vector<TreeScanner*>::const_iterator i = _treeScanners.begin();
    cv::Mat_<float> supportMap = (*i)->createSupportMap();
    i++;
    // Collect the responseMaps from the remaining trees
    for ( ; i != _treeScanners.end(); ++i)
        supportMap += (*i)->createSupportMap();
    return supportMap / _treeScanners.size();
}   // end createSupportMap


// public
int ForestScanner::getSupport( int row, int col, list<Support>& support) const
{
    int added = 0;
    const int ntrees = getNumTrees();
    for ( int i = 0; i < ntrees; ++i)
    {
        const list<Support>* tsup = _treeScanners[i]->getSupport( row, col);
        if ( tsup != NULL)
        {
            support.insert( support.end(), tsup->begin(), tsup->end());
            added += tsup->size();
        }   // end if
    }   // end for
    return added;
}   // end getSupport



// public
int ForestScanner::getSupport( int row, int col, const list<Support>* support[]) const
{
    int addedLists = 0;
    int j = 0;
    const int ntrees = getNumTrees();
    for ( int i = 0; i < ntrees; ++i)
    {
        const list<Support>* tsup = _treeScanners[i]->getSupport( row, col);
        if ( tsup != NULL)
            support[addedLists++] = tsup;
    }   // end for
    return addedLists;
}   // end getSupport



// public
cv::Mat_<byte> ForestScanner::findSupportingPoints( const cv::Rect& bbox, list<cv::Point>& spts) const
{
    const cv::Point votePoint( bbox.x + bbox.width/2, bbox.y + bbox.height/2);
    // spMap tracks locations of voting points so duplicates aren't added to spts
    cv::Mat_<byte> spMap = cv::Mat_<byte>::zeros( bbox.height, bbox.width);
    const int ntrees = getNumTrees();

    // Check each tree
    for ( int t = 0; t < ntrees; ++t)
    {
        const list<Support>* tsup = _treeScanners[t]->getSupport( votePoint.y, votePoint.x);
        if ( tsup == NULL)
            continue;

        BOOST_FOREACH( const Support& s, *tsup)
        {
            // Location voting for votePoint
            const cv::Point supPoint( votePoint.x + s.offset[0], votePoint.y + s.offset[1]);

            // Location of the voting patch must be within the given bounding box
            if ( !bbox.contains( supPoint))
                continue;

            // If already added, can ignore this supporting vote
            if ( spMap.at<byte>( supPoint.y - bbox.y, supPoint.x - bbox.x))
                continue;

            spMap.at<byte>( supPoint.y - bbox.y, supPoint.x - bbox.x) = 255;
            spts.push_back( supPoint);
        }   // end foreach
    }   // end for - trees

    return spMap;
}   // end findSupportingPoints
*/



// private virtual
void ForestScanner::process( const cv::Point& p, float pdepth, const cv::Rect& prect)
{
    // Don't process patches that overlap with the image boundary
    if (( _imgRct & prect) != prect)
        return;

    // Don't process patches that are smaller than the minimum sampling dims for
    // the feature extractors being used.
    if (( prect.width < _minSamplingDims.width) || ( prect.height < _minSamplingDims.height))
        return;

    PatchDescriptor::Ptr pd = PatchDescriptor::create();
    // Get the feature vectors for this patch - if can't get one, ignore the lot.
    const int numFXs = _fxs.size();
    for ( int i = 0; i < numFXs; ++i)
    {
        const cv::Mat_<float> fv = _fxs[i]->extract( prect);
        pd->addRowFeatureVectors(fv);
    }   // end for

    const int ntrees = getNumTrees();
    static const float minProb = 0;
    for ( int t = 0; t < ntrees; ++t)
    {
        PatchSet* treeMatches;  // vector<PatchDescriptor::Ptr>*
        const RandomTree* tree = _forest->getTree(t);
        const float prob = tree->doRegression( pd, &treeMatches);
        if ( prob >= minProb)
            addToResponseMap( p, prect, prob, pdepth, treeMatches);
    }   // end for
}   // end process



// private
void ForestScanner::addToResponseMap( const cv::Point& patchCentre, const cv::Rect& patchRect,
                                      float probability, float pdepth, const PatchSet* pmatches)
{
    cv::Mat_<float>& rmap = _responseMap;
    const cv::Rect& rct = _responseRect;

    cv::Vec2f offset;   // offset to object reference point
    cv::Vec2i pxlOffset;
    cv::Point votePoint;
    float depthWeight = 1.0f;

    const int N = pmatches->size();
    for ( int i = 0; i < N; ++i)
    {
        offset = pmatches->at(i)->getOffset();
        pxlOffset[0] = int(roundf(offset[0] * patchRect.width));
        pxlOffset[1] = int(roundf(offset[1] * patchRect.height));
        votePoint.x = patchCentre.x + pxlOffset[0];
        votePoint.y = patchCentre.y + pxlOffset[1];

        if ( _useDepthWeighting)
            depthWeight = 1.0f / (1.0f + fabsf(pdepth - _rngMap.at<float>(votePoint)));

        if ( rct.contains( votePoint))
            rmap.at<float>( votePoint) += probability * depthWeight;
    }   // end for
}   // end addToResponseMap



