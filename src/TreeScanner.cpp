#include "TreeScanner.h"
using HoughVoting::TreeScanner;
using HoughVoting::Support;
#include <cassert>
#include <cmath>
#include <ctime>
#include <iostream>
using std::cerr;
using std::endl;



TreeScanner::TreeScanner( const RandomTree* tree, const vector<FeatureExtractor::Ptr>& fxs,
                          const cv::Mat_<float>& rngMap, const cv::Mat_<byte>& mask, bool useDepthWeighting)
    : _tree(tree), _fxs(fxs), _rngMap(rngMap), _mask(mask), _useDepthWeighting( useDepthWeighting)
{
    assert( tree != NULL);
    assert( !_fxs.empty());
    FeatureExtractor::Ptr fx = *_fxs.begin();
    assert( fx != NULL);
    resetOutputMaps( fx->getImageSize());
}   // end ctor



TreeScanner::~TreeScanner()
{
    resetOutputMaps( cv::Size(0,0));
}   // end dtor



// private
void TreeScanner::resetOutputMaps( const cv::Size imgSz)
{
    const int rows = _supportMap.rows;
    const int cols = _supportMap.cols;
    for ( int i = 0; i < rows; ++i)
    {
        const list<Support>*const* srow = _supportMap.ptr<list<Support>* >(i);
        for ( int j = 0; j < cols; ++j)
        {
            if ( srow[j] != 0)
                delete srow[j];
        }   // end for
    }   // end for

    _responseMap = cv::Mat_<float>::zeros( imgSz);  // zero out response map
    _supportMap = cv::Mat_<list<Support>*>::zeros(imgSz);    // And support map
}   // end resetOutputMaps



const cv::Mat_<float> TreeScanner::scaleScan( cv::Size2f realPatchSz)
{
    resetOutputMaps( (*_fxs.begin())->getImageSize());
    // Do the scanning
    RFeatures::AdaptiveDepthPatchScanner adps( _rngMap, realPatchSz, this);
    adps.scan( _mask);
    return _responseMap;
}   // end scaleScan



const cv::Mat_<float> TreeScanner::fixedScan( cv::Size2i pxlPatchSz)
{
    const cv::Size imgSz = (*_fxs.begin())->getImageSize();
    resetOutputMaps( imgSz);

    const cv::Mat_<byte> mask = _mask;

    cv::Point p;
    cv::Rect patchRect(0,0,pxlPatchSz.width, pxlPatchSz.height);
    const int halfPatchWidth = pxlPatchSz.width/2;
    const int halfPatchHeight = pxlPatchSz.height/2;

    for ( int i = 0; i < imgSz.height; ++i)
    {
        patchRect.y = i - halfPatchHeight;
        p.y = i;
        for ( int j = 0; j < imgSz.width; ++j)
        {
            patchRect.x = j - halfPatchWidth;
            p.x = j;

            if ( mask.at<byte>(p) > 0)
                process( p, -1, patchRect);
        }   // end for - cols
    }   // end for - rows

    return _responseMap;
}   // end fixedScan



cv::Mat_<float> TreeScanner::createSupportMap() const
{
    cv::Mat_<float> smap = cv::Mat_<float>::zeros( _supportMap.size());
    const cv::Rect imgRect( 0, 0, smap.cols, smap.rows);

    const int rows = smap.rows;
    const int cols = smap.cols;
    for ( int i = 0; i < rows; ++i)
    {
        const list<Support>*const* supportRow = _supportMap.ptr<list<Support>* >( i);
        for ( int j = 0; j < cols; ++j)
        {
            const list<Support>* support = supportRow[j];
            if ( support == 0)
                continue;

            BOOST_FOREACH ( const Support& s, *support)
            {
                const cv::Point pt( j + s.offset[0], i + s.offset[1]);
                if ( imgRect.contains(pt))
                    smap.at<float>( pt) += s.probability;
            }   // end foreach
        }   // end for - cols
    }   // end for - rows

    return smap;
}   // end createSupportMap



const list<Support>* TreeScanner::getSupport( int row, int col) const
{
    assert( row >= 0 && row < _supportMap.rows);
    assert( col >= 0 && col < _supportMap.cols);
    return _supportMap.at<list<Support>* >(row, col);
}   // end getSupport



// private virtual
void TreeScanner::process( const cv::Point& p, float pdepth, const cv::Rect& patchRect)
{
    const cv::Size imSz = (*_fxs.begin())->getImageSize();
    if ( patchRect.x < 0 || patchRect.y < 0
        || patchRect.x + patchRect.width >= imSz.width || patchRect.y + patchRect.height >= imSz.height)
        return;

    // Get the feature vectors for this patch
    PatchDescriptor::Ptr pd( new PatchDescriptor);
    const int numFXs = _fxs.size();
    for ( int i = 0; i < numFXs; ++i)
    {
        const cv::Mat_<float> fv = _fxs[i]->extract( patchRect);
        pd->addRowFeatureVectors(fv);
    }   // end for

    static const float minProb = 0;
    PatchSet* treeMatches;  // vector<PatchDescriptor::Ptr>*
    const float prob = _tree->doRegression( pd, &treeMatches);
    if ( prob >= minProb)
        addToResponseMap( p, patchRect, prob, pdepth, treeMatches);
}   // end process



// private
void TreeScanner::addToResponseMap( const cv::Point& patchCentre, const cv::Rect& patchRect,
                                    float probability, float pdepth, const PatchSet* pmatches)
{
    cv::Mat_<float>& rmap = _responseMap;
    //cv::Mat_<list<Support>* >& smap = _supportMap;

    const cv::Rect responseRect( 0, 0, rmap.cols, rmap.rows);   // For convenience

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

        // Ignore votes not within the response map
        if ( responseRect.contains( votePoint))
        {
            rmap.at<float>( votePoint) += probability * depthWeight;

            // Update the list of vote support for this point
            if ( smap.at<list<Support>* >( votePoint) == 0)
                smap.at<list<Support>* >( votePoint) = new list<Support>();
            smap.at<list<Support>* >( votePoint)->push_back( Support( probability, pxlOffset));
        }   // end if
    }   // end for
}   // end addToResponseMap


