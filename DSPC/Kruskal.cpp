#include <iostream>
#include <omp.h>
#include <CL/cl.hpp>
#include <thread>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/imgproc.hpp>

#include "ImageProcessing.h"

#include <windows.h>
#include <commdlg.h>

using namespace std;
using namespace cv;
using namespace cl;

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

        vector<Edge> newEdges;
        for (int i = 0; i < edges.size(); i++)
        {
            if (edges[i].weight > threshold)
            {
                newEdges.push_back(edges[i]);
            }
        }
        edges = newEdges; // Replace the original edges vector with the filtered one
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

void mergeRegions(int start, int end, const std::vector<Edge>& edges, DisjointSet& disjointSet, cv::Mat& segmented,
                  int numCols)
{
    for (int i = start; i < end; i++)
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
}

void kruskalMergeRegionThreads(cv::Mat& segmented, const cv::Mat& input, std::vector<Edge>& edges)
{
    int numRows = input.rows;
    int numCols = input.cols;
    int numNodes = numRows * numCols;

    // Create a disjoint-set for merging segments
    DisjointSet disjointSet(numNodes);

    // Initialize the segmented image
    segmented = cv::Mat(numRows, numCols, CV_8U, cv::Scalar(0));
    std::cout << "Start merge region" << std::endl;

    // Define the number of threads to use (adjust as needed)
    int numThreads = std::thread::hardware_concurrency();

    // Calculate the number of edges each thread should process
    int edgesPerThread = edges.size() / numThreads;

    // Create a vector to hold thread objects
    std::vector<std::thread> threads;

    // Launch threads
    for (int i = 0; i < numThreads; i++)
    {
        int start = i * edgesPerThread;
        int end = (i == numThreads - 1) ? edges.size() : (i + 1) * edgesPerThread;

        threads.emplace_back(mergeRegions, start, end, std::ref(edges), std::ref(disjointSet), std::ref(segmented),
                             numCols);
    }

    // Wait for threads to complete
    for (std::thread& thread : threads)
    {
        thread.join();
    }

    std::cout << "Threaded processing complete" << std::endl;
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

void kruskalMergeRegionOpenCL(Mat& segmented, const Mat& input, vector<Edge>& edges)
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
    vector<Platform> platforms;
    Platform::get(&platforms);
    if (platforms.empty())
    {
        cout << "No OpenCL platform found" << endl;
        return;
    }

    Platform platform = platforms[0];
    vector<Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty())
    {
        cerr << "No OpenCL GPU device found" << endl;
        return;
    }

    Device device = devices[0];
    Context context(device);
    CommandQueue queue(context, device);

    const char* kernelSource = R"(
 typedef struct edge
{
    unsigned int src;
    unsigned int des;
    unsigned int weight;
} Edge;

typedef struct Pixel
{
    unsigned int x;
    unsigned int y;
    unsigned int intensity;
};

typedef struct disjointSet
{
    unsigned int* parent;
    unsigned int* rank;
    unsigned int size;
} DisjointSet;

void swap(int& a, int& b)
{
    int temp = a;
    a = b;
    b = temp;
}

int find(DisjointSet disjointSet, int v)
{
    if (v == disjointSet.parent[v])
    {
        return v;
    }
    return disjointSet.parent[v] = find(disjointSet, disjointSet.parent[v]);
}

void unionSets(int a, int b, DisjointSet disjointSet)
{
    a = find(disjointSet, a);
    b = find(disjointSet, b);

    if (a != b)
    {
        if (disjointSet.rank[a] < disjointSet.rank[b])
        {
            swap(a, b);
        }
        disjointSet.parent[b] = a;
        if (disjointSet.rank[a] == disjointSet.rank[b])
        {
            disjointSet.rank[a]++;
        }
    }
}

{
    
}


__kernel

void processEdges(__global Edge* edges,
                  __global int* segmented,
                  unsigned int numEdges,
                  unsigned int numCols)
{
    int gid = get_global_id(0);

    if (gid < numEdges)
    {
        Edge edge = edges[gid];
        DisjointSet disjointSet; // Assuming only one disjoint set in this example

        int parentA = disjointSet.parent[edge.src];
        int parentB = disjointSet.parent[edge.des];

        if (parentA != parentB)
        {
            segmented[edge.src / numCols * numCols + edge.src % numCols] = 255; // Assign segment label
            // Union operation (assuming the unionSets function is defined in the host code)
            unionSets(disjointSet, parentA, parentB);
        }else {
            segmented[edge.src / numCols * numCols + edge.src % numCols] = 0; // Assign segment label
}
    }
}
)";

    Program::Sources sources;
    sources.push_back({kernelSource, strlen(kernelSource)});
    Program program(context, sources);
    program.build({device});

    // Create buffer for data transfer
    // Buffer buffer(context, CL_MEM_READ_WRITE, sizeof(int) * numRows * numCols);
    Buffer edgesBuf(context, CL_MEM_READ_WRITE, sizeof(Edge) * edges.size());
    Buffer segmentedBuf(context, CL_MEM_READ_WRITE, sizeof(int) * numRows * numCols);

    queue.enqueueWriteBuffer(edgesBuf, CL_MEM_READ_WRITE, 0, sizeof(Edge) * edges.size(), edges.data());
    queue.enqueueWriteBuffer(segmentedBuf, CL_MEM_READ_WRITE, 0, sizeof(int) * numRows * numCols, segmented.data);

    Kernel kernel(program, "processEdges");
    kernel.setArg(0, edgesBuf);
    kernel.setArg(1, segmentedBuf);
    kernel.setArg(2, edges.size());
    kernel.setArg(3, numCols);

    int globalSize[] = {numRows, numCols};
    queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(globalSize[0], globalSize[1]));

    // Transfer data back to host
    int* output = (int*)queue.enqueueMapBuffer(segmentedBuf, CL_TRUE, CL_MAP_READ, 0, sizeof(int) * numRows * numCols);
    for (int x = 0; x < numRows; x++)
    {
        for (int y = 0; y < numCols; y++)
        {
            segmented.at<uchar>(x, y) = output[x * numCols + y];
        }
    }
    queue.enqueueUnmapMemObject(segmentedBuf, output);
    cout << "OpenCL processing complete" << endl;
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
    bool removeOutlier = false;

    kruskalImagePreprocess(inputImage, edges, removeOutlier);
    // kruskalMergeRegion(segmentedImage, inputImage, edges);
    // kruskalMergeRegionOpenMP(segmentedImage, inputImage, edges);
    double start = omp_get_wtime();
    kruskalMergeRegionThreads(segmentedImage, inputImage, edges);
    double end = omp_get_wtime();
    cout << "Thread Time: " << end - start << endl;
    // Display the segmented image
    cv::imshow("Segmented Image", segmentedImage);
    cv::waitKey(0);

    waitKey(0);
    return 0;
}

std::string WideStringToString(const wchar_t* wideStr)
{
    int bufferSize = WideCharToMultiByte(CP_UTF8, 0, wideStr, -1, nullptr, 0, NULL, NULL);
    if (bufferSize == 0)
    {
        // Failed to get the buffer size
        return "";
    }

    // Allocate a buffer to hold the converted string
    char* buffer = new char[bufferSize];

    // Convert the wide string to a narrow string (UTF-8 encoding)
    if (WideCharToMultiByte(CP_UTF8, 0, wideStr, -1, buffer, bufferSize, NULL, NULL) == 0)
    {
        // Conversion failed
        delete[] buffer;
        return "";
    }

    // Create a C++ std::string from the converted narrow string
    std::string result(buffer);

    // Clean up the allocated buffer
    delete[] buffer;

    return result;
}

// Global variable remove outlier
int removeOutlier = 0;


LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_CREATE:
        {
            // Create a "Choose Image" button
            CreateWindow(
                L"BUTTON", // Button class name
                L"Choose Image", // Button text
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, // Button style
                10, 10, 120, 30, // Button position and size
                hWnd, // Parent window handle
                (HMENU)1, // Button ID
                NULL, // Instance handle (HINSTANCE)
                NULL // Additional data
            );
            break;
        }
    case WM_COMMAND:
        {
            if (LOWORD(wParam) == 1) // Button click event
            {
                // Open a file dialog to select an image
                OPENFILENAME ofn;
                WCHAR szFile[MAX_PATH] = L"";

                ZeroMemory(&ofn, sizeof(OPENFILENAME));
                ofn.lStructSize = sizeof(OPENFILENAME);
                ofn.hwndOwner = hWnd;
                ofn.lpstrFile = szFile;
                ofn.nMaxFile = MAX_PATH;
                ofn.lpstrFilter = L"Image Files\0*.bmp;*.jpg;*.png\0All Files\0*.*\0";
                ofn.nFilterIndex = 1;
                ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
                // bool removeOutlierChecked = IsDlgButtonChecked(hWnd, 2) == BST_CHECKED;
                if (GetOpenFileName(&ofn))
                {
                    // The selected image path is now in szFile
                    // MessageBox(hWnd, szFile, L"Selected Image", MB_OK | MB_ICONINFORMATION);
                    cv::Mat inputImage = cv::imread(WideStringToString(szFile),
                                                    cv::IMREAD_GRAYSCALE);
                    cout << "input Image size: " << inputImage.cols << "x" << inputImage.rows << endl;

                    if (inputImage.empty())
                    {
                        std::cerr << "Error: Unable to load the input image." << std::endl;
                        return 1;
                    }
                    else
                    {
                        // Perform Kruskal's algorithm-based segmentation
                        cv::Mat segmentedImage;
                        vector<Edge> edges;

                        kruskalImagePreprocess(inputImage, edges, removeOutlier == 1);
                        // kruskalMergeRegion(segmentedImage, inputImage, edges);
                        kruskalMergeRegionOpenMP(segmentedImage, inputImage, edges);
                        cv::imshow("Segmented Image", segmentedImage);
                        cv::waitKey(0);
                    }
                }
            }
            else if (LOWORD(wParam) == 2) // Checkbox state change event
            {
                // Read the state of the checkbox
                bool removeOutlierChecked = IsDlgButtonChecked(hWnd, 2) == BST_CHECKED;

                // Update the removeOutlier option based on the checkbox state
                removeOutlier = removeOutlierChecked ? 1 : 0;
            }
            break;
        }
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    // Register the window class
    WNDCLASSEX wcex = {
        sizeof(WNDCLASSEX), CS_HREDRAW | CS_VREDRAW, WndProc, 0, 0, GetModuleHandle(NULL), NULL, NULL, NULL, NULL,
        L"ImageChooser", NULL
    };
    RegisterClassEx(&wcex);

    // Create the window
    HWND hWnd = CreateWindow(L"ImageChooser", L"Image Chooser", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 400,
                             200, NULL, NULL, GetModuleHandle(NULL), NULL);
    if (!hWnd)
    {
        return FALSE;
    }

    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);

    // Main message loop
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return (int)msg.wParam;
}
