#include <iostream>
#include <omp.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/imgproc.hpp>

#include "ImageProcessing.h"

using namespace std;
using namespace cv;

struct Edge
{
    int src, dest;
    int weight;
};

void kruskalImagePreprocess(Mat& input, vector<Edge>& edges, bool removeOutlier)
{
    double start = omp_get_wtime();
    cout << "Start Image Preprocess" << endl;
    int numRows = input.rows;
    int numCols = input.cols;
    int numNodes = numRows * numCols;

    // Create a vector to store edges
    // std::vector<Edge> edges;
    // Populate the edges with intensity differences as weights
    for (int x = 0; x < numRows; ++x)
    {
        for (int y = 0; y < numCols; ++y)
        {
            int nodeA = x * numCols + y;
            if (x < numRows - 1)
            {
                // Check lower neighbor
                int nodeB = (x + 1) * numCols + y;
                int weight = std::abs(input.at<uchar>(x, y) - input.at<uchar>(x + 1, y));
                edges.push_back({nodeA, nodeB, weight});
            }
            if (y < numCols - 1)
            {
                // Check right neighbor
                int nodeB = x * numCols + (y + 1);
                int weight = std::abs(input.at<uchar>(x, y) - input.at<uchar>(x, y + 1));
                edges.push_back({nodeA, nodeB, weight});
            }
        }
    }
    double pptime = omp_get_wtime();
    cout << "Preprocess Time: " << pptime - start << endl << endl;
    if (removeOutlier)
    {
        cout << "Start eliminate outlier" << endl;
        int sum = 0;
        for (int i = 0; i < edges.size(); i++)
        {
            sum += edges[i].weight;
        }
        cout << "Sum: " << sum << endl;
        int average = sum / edges.size();
        int stdDev = 0;
        for (int i = 0; i < edges.size(); i++)
        {
            stdDev += (edges[i].weight - average) * (edges[i].weight - average);
        }
        cout << "Standard Deviation: " << stdDev << endl;
        stdDev = sqrt(stdDev / edges.size());
        int threshold = average + stdDev;
        cout << "Threshold: " << threshold << endl;
        cout << "Removing outlier for size of: " << edges.size() << endl;
        // Remove edges with weight less than or equal to threshold

        for (int i = 0; i < edges.size(); i++)
        {
            if (edges[i].weight <= threshold)
            {
                edges.erase(edges.begin() + i);
            }
        }
    }
    double emtime = omp_get_wtime() - pptime;
    cout << "Eliminate Time: " << emtime << endl << endl;
    cout << "Start sorting" << endl;
    // Sort edges by weight
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b)
    {
        return a.weight < b.weight;
    });
    cout << "Sort Time: " << omp_get_wtime() - emtime << endl << endl;
}

void kruskalMergeRegion(Mat& segmented, const Mat& input, vector<Edge>& edges)
{
    int numRows = input.rows;
    int numCols = input.cols;
    int numNodes = numRows * numCols;
    // Create a disjoint-set for merging segments
    DisjointSet disjointSet(numNodes);

    // Initialize the segmented image
    segmented = cv::Mat(numRows, numCols, CV_8U, cv::Scalar(0));
    cout << "Start merge region" << endl;
    double start = omp_get_wtime();
    // Merge segments using Kruskal's algorithm
    for (const Edge& edge : edges)
    {
        int parentA = disjointSet.find(edge.src);
        int parentB = disjointSet.find(edge.dest);

        if (parentA != parentB)
        {
            segmented.at<uchar>(edge.src / numCols, edge.src % numCols) = 255; // Assign segment label
            disjointSet.unionSets(parentA, parentB);
        }
    }
    double end = omp_get_wtime();
    cout << "Normal Time: " << end - start << endl << endl;
}

void kruskalMergeRegionOpenMP(Mat& segmented, const Mat& input, vector<Edge>& edges)
{
    int numRows = input.rows;
    int numCols = input.cols;
    int numNodes = numRows * numCols;
    // Create a disjoint-set for merging segments
    DisjointSet disjointSet(numNodes);

    // Initialize the segmented image
    segmented = cv::Mat(numRows, numCols, CV_8U, cv::Scalar(0));
    cout << "Start merge region" << endl;
    // Merge segments using Kruskal's algorithm
    double start = omp_get_wtime();
#pragma omp parallel for shared(disjointSet, segmented)
    for (int i = 0; i < edges.size(); i++)
    {
        Edge edge = edges[i];
        int parentA = disjointSet.find(edge.src);
        int parentB = disjointSet.find(edge.dest);

        if (parentA != parentB)
        {
            segmented.at<uchar>(edge.src / numCols, edge.src % numCols) = 255; // Assign segment label
            disjointSet.unionSets(parentA, parentB);
        }
    }
    double end = omp_get_wtime();
    cout << "OpenMP Time: " << end - start << endl;
}

int main()
{
    // "C:\\Users\\TYH\\source\\repos\\DSPC\\x64\\Debug\\lena.png"
    // "C:\\Users\\TYH\\Downloads\\SNAPSHOT.png"
    cv::Mat inputImage = cv::imread("C:\\Users\\TYH\\Downloads\\SNAPSHOT.png",
                                    cv::IMREAD_GRAYSCALE);
    cout << "input Image size: " << inputImage.cols << "x" << inputImage.rows << endl;

    if (inputImage.empty())
    {
        std::cerr << "Error: Unable to load the input image." << std::endl;
        return 1;
    }

    // Perform Kruskal's algorithm-based segmentation
    cv::Mat segmentedImage;
    vector<Edge> edges;
    bool removeOutlier = true;

    kruskalImagePreprocess(inputImage, edges, removeOutlier);
    kruskalMergeRegion(segmentedImage, inputImage, edges);
    kruskalMergeRegionOpenMP(segmentedImage, inputImage, edges);

    // Display the segmented image
    cv::imshow("Segmented Image", segmentedImage);
    cv::waitKey(0);

    waitKey(0);
    return 0;
}
