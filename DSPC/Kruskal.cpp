#include <iostream>

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

void kruskalArraySegmentation(Mat& segmented, const Mat& input)
{
    int numRows = input.rows;
    int numCols = input.cols;
    int numNodes = numRows * numCols;

    // Maximum number of possible edges (assuming 4-connectivity)
    const int maxEdges = 2 * numRows * numCols - numRows - numCols;

    // Create an array to store edges
    Edge* edges = new Edge[maxEdges];

    int edgeCount = 0; // Track the number of edges

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
                edges[edgeCount++] = {nodeA, nodeB, weight};
            }
            if (y < numCols - 1)
            {
                // Check right neighbor
                int nodeB = x * numCols + (y + 1);
                int weight = std::abs(input.at<uchar>(x, y) - input.at<uchar>(x, y + 1));
                edges[edgeCount++] = {nodeA, nodeB, weight};
            }
        }
    }

    // Sort edges by weight (you can use a custom sorting function)
    // Implement a sorting algorithm or use std::sort if you prefer
    // std::sort(edges, edges + edgeCount, [](const Edge& a, const Edge& b) { return a.weight < b.weight; });

    // Create an array for the disjoint-set data structure
    int* parent = new int[numNodes];

    // Initialize the parent array
    for (int i = 0; i < numNodes; ++i)
    {
        parent[i] = i;
    }

    // Initialize the segmented image
    segmented = cv::Mat(numRows, numCols, CV_8U, cv::Scalar(0));

    // Merge segments using Kruskal's algorithm
    for (int i = 0; i < edgeCount; ++i)
    {
        int parentA = parent[edges[i].src];
        int parentB = parent[edges[i].dest];

        if (parentA != parentB)
        {
            segmented.at<uchar>(edges[i].src / numCols, edges[i].src % numCols) = 255; // Assign segment label
            parent[parentB] = parentA; // Merge segments
        }
    }

    // Clean up allocated memory
    delete[] edges;
    delete[] parent;
}

// Function to perform image segmentation using Kruskal's algorithm
void kruskalOpenMPSegmentation(cv::Mat& segmented, const cv::Mat& input)
{
    int numRows = input.rows;
    int numCols = input.cols;
    int numNodes = numRows * numCols;

    // Create a vector to store edges
    std::vector<Edge> edges;
    // Populate the edges with intensity differences as weights
#pragma omp parallel for shared(edges)
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
#pragma omp critical
                {
                    edges.push_back({nodeA, nodeB, weight});
                }
            }
            if (y < numCols - 1)
            {
                // Check right neighbor
                int nodeB = x * numCols + (y + 1);
                int weight = std::abs(input.at<uchar>(x, y) - input.at<uchar>(x, y + 1));
#pragma omp critical
                {
                    edges.push_back({nodeA, nodeB, weight});
                }
            }
        }
    }

    // Sort edges by weight
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b)
    {
        return a.weight < b.weight;
    });

    // Create a disjoint-set for merging segments
    DisjointSet disjointSet(numNodes);

    // Initialize the segmented image
    segmented = cv::Mat(numRows, numCols, CV_8U, cv::Scalar(0));

    // Merge segments using Kruskal's algorithm
#pragma omp parallel for shared(disjointSet, segmented)
    for (int i = 0; i < edges.size(); i++)
    {
        Edge edge = edges[i];
        int parentA = disjointSet.find(edge.src);
        int parentB = disjointSet.find(edge.dest);

        if (parentA != parentB)
        {
#pragma omp critical
            {
                segmented.at<uchar>(edge.src / numCols, edge.src % numCols) = 255; // Assign segment label
                disjointSet.unionSets(parentA, parentB);
            }
        }
    }
}

// Function to perform image segmentation using Kruskal's algorithm
void kruskalSegmentation(cv::Mat& segmented, const cv::Mat& input)
{
    int numRows = input.rows;
    int numCols = input.cols;
    int numNodes = numRows * numCols;

    // Create a vector to store edges
    std::vector<Edge> edges;

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

    int sum = 0;
    for (int i = 0; i < edges.size(); i++)
    {
        sum += edges[i].weight;
    }
    int average = sum / edges.size();
    int stdDev = 0;
    for (int i = 0; i < edges.size(); i++)
    {
        stdDev += (edges[i].weight - average) * (edges[i].weight - average);
    }
    stdDev = sqrt(stdDev / edges.size());
    int threshold = average + stdDev;

    // Remove edges with weight less than or equal to threshold
    for (int i = 0; i < edges.size(); i++)
    {
        if (edges[i].weight <= threshold)
        {
            edges.erase(edges.begin() + i);
        }
    }

    // Sort edges by weight
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b)
    {
        return a.weight < b.weight;
    });

    // Create a disjoint-set for merging segments
    DisjointSet disjointSet(numNodes);

    // Initialize the segmented image
    segmented = cv::Mat(numRows, numCols, CV_8U, cv::Scalar(0));

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
}

int main()
{
    Mat image = imread("C:\\Users\\TYH\\source\\repos\\DSPC\\x64\\Debug\\lena.png");
    cout << "Image size: " << image.cols << "x" << image.rows << endl;
    int edgeCount;
    int mstCount;
    // Edge* edges = Edge::fromImage(image, edgeCount);
    // Edge* mst = Edge::formKruskal(edges, image.cols, image.rows, mstCount);
    // Edge* groupMST(mst, mstCount);
    // drawMST(image, mst, mstCount);
    // Pixel** regions = segmentOpenMP(mst, edgeCount, mstCount);
    cv::Mat inputImage = cv::imread("C:\\Users\\TYH\\source\\repos\\DSPC\\x64\\Debug\\lena.png",
                                    cv::IMREAD_GRAYSCALE);

    if (inputImage.empty())
    {
        std::cerr << "Error: Unable to load the input image." << std::endl;
        return 1;
    }

    // Perform Kruskal's algorithm-based segmentation
    cv::Mat segmentedImage;
    kruskalSegmentation(segmentedImage, inputImage);

    // Display the segmented image
    cv::imshow("Segmented Image", segmentedImage);
    cv::waitKey(0);

    waitKey(0);
    return 0;
}
