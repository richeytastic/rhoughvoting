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
#include <cstdlib>
#include <algorithm>
#include <iostream>
using std::cerr;
using std::endl;

#include "TestParamGenerator.h"
using HoughVoting::TestParamGenerator;


/*************************************************************/
/***************** TEST FOR OFFSET REGRESSION ****************/
/*************************************************************/
double calc2DOffsetSSD( const PatchSet& s)
{
    // Get the mean offset vector
    const int sz = s.size();
    cv::Vec2f u(0,0);
    for ( int i = 0; i < sz; ++i)
        u += s[i]->getOffset();
    u /= sz;

    double ssd = 0;
    for ( int i = 0; i < sz; ++i)
    {
        const cv::Vec2f& offset = s[i]->getOffset();
        ssd += pow( offset[0] - u[0], 2) + pow( offset[1] - u[1], 2);
    }   // end for
    return ssd;
}   // end calc2DOffsetSSD


double calc2DOffsetDistMean(const PatchSet& setA, const PatchSet& setB)
{
    const double distA = calc2DOffsetSSD( setA);
    const double distB = calc2DOffsetSSD( setB);
	return (distA + distB) / (setA.size() + setB.size()); 
}   // end calc2DOffsetDistMean
/*************************************************************/
/*************************************************************/
/*************************************************************/



/*************************************************************/
/****************** TEST FOR CLASSIFICATION ******************/
/*************************************************************/
// Maximised when numNeg very different to numPos
double calcNegEntropy( double numNeg, double numPos)
{
	// negative entropy: sum_i p_i*log(p_i)
	double nentropy = 0;
    const double p = numNeg / (numNeg + numPos);
    if ( p > 0)
        nentropy = p*log(p) + (1.-p)*log(1.-p);
    return nentropy;
}   // end calcNegEntropy

// Returns a negative value. Values closer to zero indicate better discriminative power.
double calcInfGain( int posA, int negA, int posB, int negB)
{
	const double sizeA = posA + negA;
    const double sizeB = posB + negB;
	return (sizeA * calcNegEntropy( negA, posA) + sizeB * calcNegEntropy( negB, posB))
          / (sizeA + sizeB);
}   // end calcInfGain
/*************************************************************/
/*************************************************************/
/*************************************************************/



typedef struct RandomTree::Node
{
    Node( RandomTree*, const TrainSet&, int depth);
    Node( RandomTree*, int depth);
    ~Node();

    int depth;  // Node depth in the tree
    RandomTree* rtree;
    Node* leftNode;
    Node* rightNode;

    float threshold;
    int testChannel;    // Feature vector chosen to break positive and negative sets on
    int testIndices[2]; // Indices into chosen feature vector

    float fgProbability;
    PatchSet patches;    // Each has decriptor (cv::Mat_<float>) and offset (cv::Vec2f)

    void save( ostream&, bool ioFeatures) const; // Recursively save out this node and all children
    void load( istream&, bool ioFeatures);   // Recursively load this node and all children

    float findMatch( const PatchDescriptor::Ptr, PatchSet**);

private:
    bool optimiseTestParams( const TrainSet&, TrainSet& setA, TrainSet& setB);
    void makeLeaf( const TrainSet&);
} Node;  // end struct



float Node::findMatch( const PatchDescriptor::Ptr pd, PatchSet** matches)
{
    if ( testChannel == -1) // Is leaf so found result
    {
        *matches = &patches;
        return fgProbability;
    }   // end if

    const float f1 = pd->val( testChannel, testIndices[0]);
    const float f2 = pd->val( testChannel, testIndices[1]);
    if ( leftNode != 0 && ( f1 - f2 ) < threshold)  // go left
        return leftNode->findMatch( pd, matches);
    else if ( rightNode != 0)
        return rightNode->findMatch( pd, matches);
}   // end findMatch



void Node::load( istream& is, bool readFeatures)
{
    char leftChild, rightChild;

    string ln;
    std::getline( is, ln);
    std::istringstream iss(ln);
    iss >> depth >> testChannel >> testIndices[0] >> testIndices[1] >> threshold >> leftChild >> rightChild;

    assert( leftChild == '_' || leftChild == 'L');
    assert( rightChild == '_' || rightChild == 'R');
    assert( testChannel >= -1 && testChannel <= 31);

    if ( testChannel == -1) // Leaf node
    {
        assert( leftNode == 0 && rightNode == 0);
        int numPatches;
        bool readFeatures;
        string leafHeader;
        std::getline( is, leafHeader);
        std::istringstream iss1(leafHeader);
        iss1 >> fgProbability >> numPatches >> readFeatures;

        assert( numPatches >= 0);
        if ( numPatches > 0)
        {
            patches.resize(numPatches);
            for ( int i = 0; i < numPatches; ++i)
            {
                PatchDescriptor::Ptr patch( new PatchDescriptor());
                if ( readFeatures)
                    is >> *patch;
                else
                {
                    cv::Vec2f v;
                    is >> v[0] >> v[1];
                    patch->setOffset(v);
                    std::getline(is,ln);    // Discard end of line
                }   // end else
                patches[i] = patch;
            }   // end for
        }   // end if

        rtree->_leaves.push_back(this);
    }   // end if

    if ( leftChild == 'L')
    {
        // Grow left
        leftNode = new Node( rtree, depth+1);
        leftNode->load( is, readFeatures);
    }   // end if

    if ( rightChild == 'R')
    {
        // Grow right
        rightNode = new Node( rtree, depth+1);
        rightNode->load( is, readFeatures);
    }   // end if
}   // end load



void Node::save( ostream& os, bool writeFeatures) const
{
    const char leftChild = (leftNode == 0) ? '_' : 'L';
    const char rightChild = (rightNode == 0) ? '_' : 'R';

    using std::endl;
    // testChannel and indices are always -1 for leaf nodes
    os << depth << " " << testChannel << " " << testIndices[0] << " " << testIndices[1] << " " << threshold
       << " " << leftChild << " " << rightChild << endl;  // Encode child node structure

    if ( testChannel == -1)
    {
        // This is a leaf node so write out the probability and the patchset.
        // Number of patches and whether patch features included.
        os << fgProbability << " " << patches.size() << " " << writeFeatures << endl;
        BOOST_FOREACH ( const PatchDescriptor::Ptr patch, patches)
        {
            if ( writeFeatures)
                os << *patch << endl;   // Feature information is not needed for classification purposes
            else
            {
                const cv::Vec2f& v = patch->getOffset();
                os << v[0] << " " << v[1] << endl;
            }   // end else
        }   // end for
    }   // end if
    else
    {
        // Keep going left if available
        if ( leftChild == 'L')
            leftNode->save( os, writeFeatures);

        // Done going left; now go right
        if ( rightChild == 'R')
            rightNode->save( os, writeFeatures);
    }   // end else
}   // end save



Node::Node( RandomTree* rt, const TrainSet& trainSet, int dpth)
    : depth(dpth), rtree( rt), leftNode(0), rightNode(0), threshold(0)
{
    boost::this_thread::interruption_point();
    testChannel = -1;
    testIndices[0] = -1;
    testIndices[1] = -1;

    assert( trainSet.size() == 2);
    if ( depth >= rtree->_maxDepth || trainSet[1].empty())  // Only neg patches are left or maximum depth is reached
        this->makeLeaf( trainSet);
    else
    {	
        // Find optimal test parameters given a randomised measurement mode
        TrainSet leftSet, rightSet;
        if ( optimiseTestParams( trainSet, leftSet, rightSet))
        {
            const int leftCount = leftSet[0].size() + leftSet[1].size();
            const int rightCount = rightSet[0].size() + rightSet[1].size();

            // Go left
            // If enough patches remain, continue growing left subtree else stop
            if ( leftCount > rtree->_minSamplesPerLeaf)
                this->leftNode = new Node( rtree, leftSet, depth+1);
            else
            {
                this->leftNode = new Node( rtree, depth+1);
                this->leftNode->makeLeaf( leftSet);
            }   // end else

            // Go right
            // If enough patches remain, continue growing right subtree else stop
            if ( rightCount > rtree->_minSamplesPerLeaf)
                this->rightNode = new Node( rtree, rightSet, depth+1);
            else
            {
                this->rightNode = new Node( rtree, depth+1);
                this->rightNode->makeLeaf( rightSet);
            }   // end else

            rtree->updateProgress( depth, -1, -1, testChannel); // Update depth stats
        }   // end if
        else // Could not find split (only invalid one leave split)
            this->makeLeaf( trainSet);
    }   // end else
}   // end Node::Node



Node::Node( RandomTree* rt, int dpth)
    : depth(dpth), rtree( rt), leftNode(0), rightNode(0), threshold(0)
{
    boost::this_thread::interruption_point();
    testChannel = -1;
    testIndices[0] = -1;
    testIndices[1] = -1;
}   // end Node::Node


Node::~Node()
{
    if ( leftNode != 0)
        delete leftNode;
    if ( rightNode != 0)
        delete rightNode;
}   // end dtor


// Create leaf node from patches 
void Node::makeLeaf( const TrainSet& leafSet)
{
    const int numFGs = leafSet[1].size();
    const int numBGs = leafSet[0].size();

    const float pnratio = float(rtree->_totalFGs) / rtree->_totalBGs;
	fgProbability = float(numFGs) / (pnratio*float(numBGs) + float(numFGs));

    // Store the positive PatchDescriptor objects in this node
    const PatchSet& posSet = leafSet[1];
    patches.insert( patches.end(), posSet.begin(), posSet.end());

    rtree->updateProgress( depth, numFGs, numBGs, -1);
    rtree->_leaves.push_back(this);
}   // end Node::makeLeaf



// Calculate the difference between the values from two locations within feature vector fvIdx
// for each patch, for each class. Returns the maximum difference found (i.e. from a certain patch).
float evaluateTest( const TrainSet& trainSet, int fvIdx, int i0, int i1, float &vmin, vector<vector<float> >& featureDiffs)
{
    vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    const int numClasses = trainSet.size();
    assert( numClasses == 2);
    for ( int j = 0; j < numClasses; ++j)
    {
        const PatchSet& patches = trainSet[j];
        const int numClassPatches = patches.size();
		for ( int i = 0; i < numClassPatches; ++i)
        {
            const float d = patches[i]->val( fvIdx, i0) - patches[i]->val( fvIdx, i1);
			featureDiffs[j].push_back( d);
            // find min/max values for threshold
            if ( d < vmin) vmin = d;
            if ( d > vmax) vmax = d;
		}   // end for
	}   // end foreach (classes==2)

    return vmax - vmin;
}   // end evaluateTest



void splitTrainSet( const PatchSet& pSet, const vector<float>& featureDiffs, float T, PatchSet& childNodeA, PatchSet& childNodeB)
{
    const int n = featureDiffs.size();
    for ( int i = 0; i < n; ++i)
    {
        const PatchDescriptor::Ptr pf = pSet[i];
        if ( featureDiffs[i] < T)
            childNodeA.push_back( pf);
        else
            childNodeB.push_back( pf);
    }   // end for
}   // end splitTrainSet



bool Node::optimiseTestParams( const TrainSet& trainSet, TrainSet& setA, TrainSet& setB)
{
    rlib::Random* random = &rtree->_random; // This tree's random number generator

    // Set measure mode for split: 0 - classification, 1 - offset regression
    int measureMode = 1;    // Offset regression by default
    // If there's still >= 5% negatives in the training set and this node's depth is not too deep,
    // randomly select between both options.
    const int numNeg = trainSet[0].size();
    const int numPos = trainSet[1].size();
    if ( (float(numNeg) / float(numNeg+numPos) >= 0.05) && (depth < rtree->_maxDepth-2))
        measureMode = random->getRandomInt() % 2;

    const int numLabs = trainSet.size();
    const PatchDescriptor::Ptr samplePD = trainSet[1][0];
    const int numFVs = samplePD->getNumFeatureVectors();

	bool found = false;
	double bestDist = -DBL_MAX; // to maximize

    // Find best test of ITER iterations unless descriptor is small enough for a comprehensive search
    // of the test parameters at this node. ITER sets a hard limit on search complexity through the
    // different feature vectors of a patch descriptor and the indices into those feature vectors.
    static const int ITER = 2000;

    // Get the range of each feature vector
    int paramRanges[numFVs];
    for ( int i = 0; i < numFVs; ++i)
    {
        paramRanges[i] = samplePD->getFeatureVectorSize(i);
        assert( paramRanges[i] > 1);
    }   // end for

    TestParamGenerator tpg( numFVs, paramRanges, ITER);

    int fvIdx = 0;  // Feature vector index
    int i0, i1; // Test indices into feature vector fvIdx

    // Select the feature that (when comparing two values from it) gives the best ability to
    // discriminate between the positive or negative class, or to maximise the similarity
    while (( fvIdx = tpg.nextParams( i0, i1)) < numFVs)
    {
        boost::this_thread::interruption_point();
        // i0 and i1 are now a pair of random indices into randomly selected feature vector fvIdx.

		// Compute value for each patch and find the largest difference between 2 patches over all objects of all (2) classes.
	    vector<vector<float> > featureDiffs(numLabs); // temporary data for finding best test
        float vmin = 0;
		const float diffRange = evaluateTest( trainSet, fvIdx, i0, i1, vmin, featureDiffs);

        // Find best threshold over 10 iterations
        for ( int j = 0; j < 10; ++j)
        { 
            const float T = (random->getRandom() * diffRange) + vmin; // Generate random threshold

            // Put +ve and -ve patch exemplars together into the prospective split sets
            // based on whether the chosen feature difference is <= or > T

            // Split training data into two temporary sets A,B accroding to threshold T
            TrainSet tmpA( numLabs);    // Prospective split set A (left node)
            TrainSet tmpB( numLabs);    // Prospective split set B (right node)
            int tmpAsz = 0, tmpBsz = 0;
            for ( int i = 0; i < numLabs; ++i)
            {
                splitTrainSet( trainSet[i], featureDiffs[i], T, tmpA[i], tmpB[i]);   // Split the class set
                tmpAsz += tmpA[i].size();
                tmpBsz += tmpB[i].size();
            }   // end for

            // Don't allow empty set splits
            if ( tmpAsz == 0 || tmpBsz == 0)
                continue;

            // Measure quality of split according to measureMode
            double tmpDist = 0;
            switch ( measureMode)
            {
                case 0:
                    tmpDist = calcInfGain( tmpA[1].size(), tmpA[0].size(), tmpB[1].size(), tmpB[0].size());
                    break;
                case 1:
                    tmpDist = -calc2DOffsetDistMean( tmpA[1], tmpB[1]);
                    break;
                default:
                    assert(false);
            }   // end switch

            // Take binary test with best split
            if  (tmpDist > bestDist)
            {
                found = true;
                bestDist = tmpDist;

                testChannel = fvIdx;
                testIndices[0] = i0;
                testIndices[1] = i1;
                threshold = T;

                setA = tmpA;
                setB = tmpB;
            }   // end if
        } // end for j
	} // end while

	// Return true if valid test is found. Test is invalid if only empty set splits found.
	return found;
}   // end optimiseTestParams




RandomTree::RandomTree( int minSamplesPerLeaf, int maxDepth)
    : _minSamplesPerLeaf(minSamplesPerLeaf), _maxDepth(maxDepth), _root(0), _ioFeatures(false)
{
}   // end ctor



RandomTree::~RandomTree()
{
    if ( _root != 0)
        delete _root;
}   // end dtor



void RandomTree::grow( const TrainSet* ts, int randInit)
{
    const TrainSet& trainSet = *ts;
    const int numSets = trainSet.size();
    assert( numSets == 2);  // Only BG and FG objects allowed for now

    // Ensure the number of feature extractors for the positive and negative sets are equal
    int numFXs = -1;
    for ( int c = 0; c < numSets; ++c)
    {
        const int numFXs = trainSet[c][0]->getNumFeatureVectors();
        if ( c == 0)
            _numFXs = numFXs;
        assert( _numFXs == numFXs);
    }   // end for

    _totalBGs = trainSet[0].size();
    _totalFGs = trainSet[1].size();

    _depthStats.clear();
    _totalPatchesParsed = 0;
    _random = rlib::Random( randInit);

    ProgressDelegate* tup = getTrainingProgressUpdater();   // Tree
    if ( tup != NULL)
        tup->updateProgress( 0);

    _leaves.clear();
    _root = new Node( this, trainSet, 0);   // Recursively grow the tree

    assert( _totalPatchesParsed == _totalFGs + _totalBGs);
}   // end grow



void RandomTree::updateProgress( int nodeDepth, int numFGs, int numBGs, int featChannel)
{
    while ( nodeDepth >= _depthStats.size())
    {
        DepthStats ds;
        ds.numNodes = 0;
        ds.numLeaves = 0;
        ds.featureCounts.resize( _numFXs);
        _depthStats.push_back(ds);
    }   // end while

    _depthStats[nodeDepth].numNodes++;
    if ( featChannel >= 0)
    {
        // Record the feature used to split on at this level
        _depthStats[nodeDepth].featureCounts[featChannel]++;
    }   // end if
    else    // Otherwise, this is a leaf node
    {
        _depthStats[nodeDepth].numLeaves++;
        ProgressDelegate* tup = getTrainingProgressUpdater();   // Tree
        if ( tup != NULL)
        {
            _totalPatchesParsed += numFGs + numBGs;
            tup->updateProgress( float(_totalPatchesParsed) / (_totalFGs + _totalBGs));
        }   // end if
    }   // end if
}   // end updateProgress



float RandomTree::doRegression( const PatchDescriptor::Ptr pd, PatchSet** matches) const
{
    // Could do this in a loop but recursion is more instructive.
    return _root->findMatch( pd, matches);
}   // end doRegression



void RandomTree::setReadWriteFeatures( bool enable)
{
    _ioFeatures = enable;
}   // end setReadWriteFeatures



ostream& HoughVoting::operator<<( ostream& os, const RandomTree& tree)
{
    // Write out tree in left child order
    assert( tree._root != NULL);
    tree._root->save( os, tree._ioFeatures);
    return os;
}   // end operator<<



istream& HoughVoting::operator>>( istream& is, RandomTree& tree)
{
    tree._root = new Node( &tree, 0);
    tree._leaves.clear();
    tree._root->load( is, tree._ioFeatures);
    return is;
}   // end operator>>



cv::Mat_<float> RandomTree::makeLeafProbMap( int maxDimPxls) const
{
    float maxWidth = 0;
    float maxHeight = 0;

    // Find the largest offset
    BOOST_FOREACH ( const Node* node, _leaves)
    {
        BOOST_FOREACH ( const PatchDescriptor::Ptr p, node->patches)
        {
            const cv::Vec2f& offset = p->getOffset();
            maxWidth = std::max<float>( maxWidth, fabsf(offset[0]));
            maxHeight = std::max<float>( maxHeight, fabsf(offset[1]));
        }   // end foreach
    }   // end foreach

    cv::Size imgSz;
    if ( maxHeight > maxWidth)
        imgSz = cv::Size( maxWidth/maxHeight * maxDimPxls, maxDimPxls);
    else
        imgSz = cv::Size( maxDimPxls, maxHeight/maxWidth * maxDimPxls);

    cv::Mat_<float> img = cv::Mat_<float>::zeros(imgSz);
    const int hrows = imgSz.height/2;
    const int hcols = imgSz.width/2;

    BOOST_FOREACH ( const Node* node, _leaves)
    {
        const float prob = node->fgProbability;
        BOOST_FOREACH ( const PatchDescriptor::Ptr p, node->patches)
        {
            const cv::Vec2f& offset = p->getOffset();   // Offset from instance centre to patch centre
            img.at<float>( std::min<int>( hrows + hrows * offset[1]/maxHeight, imgSz.height-1),
                           std::min<int>( hcols + hcols * offset[0]/maxWidth, imgSz.width-1)) += prob;
        }   // end foreach
    }   // end foreach

    return img;
}   // end makeLeafProbMap

