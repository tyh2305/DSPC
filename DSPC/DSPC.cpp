#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <ctime>

using namespace cv;
using namespace std;

// // Structure to represent an edge in the graph
// struct Edge
// {
//     int u, v; // Vertices connected by the edge
//     float weight; // Weight of the edge
// };
//
// // Structure to represent a disjoint-set element
// struct DisjointSetElement
// {
//     int parent;
//     int rank;
// };
//
// // Function to find the set to which an element belongs (with path compression)
// int findSet(vector<DisjointSetElement>& sets, int i)
// {
//     if (i != sets[i].parent)
//     {
//         sets[i].parent = findSet(sets, sets[i].parent);
//     }
//     return sets[i].parent;
// }
//
// // Function to perform union of two sets (with union by rank)
// void unionSets(vector<DisjointSetElement>& sets, int i, int j)
// {
//     int root1 = findSet(sets, i);
//     int root2 = findSet(sets, j);
//     if (root1 != root2)
//     {
//         if (sets[root1].rank < sets[root2].rank)
//         {
//             sets[root1].parent = root2;
//         }
//         else if (sets[root1].rank > sets[root2].rank)
//         {
//             sets[root2].parent = root1;
//         }
//         else
//         {
//             sets[root2].parent = root1;
//             sets[root1].rank++;
//         }
//     }
// }
//
// void kruskal(Mat image, vector<Edge> edges)
// {
//     int numVertices = image.rows * image.cols;
//     // Initialize disjoint-set data structure
//     vector<DisjointSetElement> sets(numVertices);
//     for (int i = 0; i < numVertices; ++i)
//     {
//         sets[i].parent = i;
//         sets[i].rank = 0;
//     }
//
//     // Initialize variables for segmentation
//     vector<Vec3b> segmentColors(numVertices);
//     Mat segmentedImage(image.size(), CV_8UC3, Scalar(0, 0, 0));
//     int numComponents = numVertices;
//
//     // Initialize disjoint-set data structure
//     for (int i = 0; i < numVertices; ++i)
//     {
//         sets[i].parent = i;
//         sets[i].rank = 0;
//         // Initialize segment colors with random colors
//         segmentColors[i] = Vec3b(rand() & 255, rand() & 255, rand() & 255);
//     }
//
//     clock_t start_kruskal = clock();
//     // Apply Kruskal's algorithm to find the MST and segment the image
//     for (const Edge& edge : edges)
//     {
//         int root1 = findSet(sets, edge.u);
//         int root2 = findSet(sets, edge.v);
//         if (root1 != root2)
//         {
//             unionSets(sets, root1, root2);
//             // Assign the color of the representative vertex to the entire segment
//             Vec3b color = segmentColors[root1];
//             for (int y = 0; y < image.rows; ++y)
//             {
//                 for (int x = 0; x < image.cols; ++x)
//                 {
//                     int vertex = y * image.cols + x;
//                     if (findSet(sets, vertex) == root1)
//                     {
//                         segmentedImage.at<Vec3b>(y, x) = color;
//                     }
//                 }
//             }
//             numComponents--;
//             if (numComponents == 1)
//             {
//                 break; // Stop when all pixels are in one component
//             }
//         }
//     }
//
//     // Display and save the segmented result
//     imshow("Segmented Image", segmentedImage);
//     imwrite("segmented_image.jpg", segmentedImage);
// }
// class ImageGraphEdge
// {
//     int x;
//     int y;
//     int w;
//
// public:
//     ImageGraphEdge(int x, int y, int w)
//     {
//         this->x = x;
//         this->y = y;
//         this->w = w;
//     }
// };
//
// class ImageGraph
// {
//     vector<ImageGraphEdge> edgelist;
//
// public:
//     ImageGraph fromMat(Mat image)
//     {
//         for (int y = 0; y < image.rows; ++y)
//         {
//             for (int x = 0; x < image.cols; ++x)
//             {
//                 edgelist.push_back(ImageGraphEdge(x, y, static_cast<float>(image.at<uchar>(y, x))));
//             }
//         }
//     }
//
//     void visualizeMat(Mat image)
//     {
//         cout << "Image Graph" << endl;
//         for (int y = 0; y < image.rows; ++y)
//         {
//             // for (int x = 0; x < image.cols; ++x)
//             // {
//             //     cout << "|---";
//             // }
//             // cout << "|" << endl;
//             // for (int x = 0; x < image.cols; ++x)
//             // {
//             //     cout << "|" << static_cast<float>(image.at<uchar>(y, x));
//             // }
//             // cout << "|" << endl;
//         }
//     }
// };
//
// class DSU
// {
//     int* parent;
//     int* rank;
//
// public:
//     DSU(int n)
//     {
//         parent = new int[n];
//         rank = new int[n];
//
//         for (int i = 0; i < n; i++)
//         {
//             parent[i] = -1;
//             rank[i] = 1;
//         }
//     }
//
//     // Find function
//     int find(int i)
//     {
//         if (parent[i] == -1)
//             return i;
//
//         return parent[i] = find(parent[i]);
//     }
//
//     // Union function
//     void unite(int x, int y)
//     {
//         int s1 = find(x);
//         int s2 = find(y);
//
//         if (s1 != s2)
//         {
//             if (rank[s1] < rank[s2])
//             {
//                 parent[s1] = s2;
//             }
//             else if (rank[s1] > rank[s2])
//             {
//                 parent[s2] = s1;
//             }
//             else
//             {
//                 parent[s2] = s1;
//                 rank[s1] += 1;
//             }
//         }
//     }
// };
//
// class Graph
// {
//     vector<vector<int>> edgelist;
//     int V;
//
// public:
//     Graph(int V) { this->V = V; }
//
//     // Function to add edge in a graph
//     void addEdge(int x, int y, int w)
//     {
//         edgelist.push_back({w, x, y});
//     }
//
//     void kruskals_mst()
//     {
//         // Sort all edges
//         sort(edgelist.begin(), edgelist.end());
//
//         // Initialize the DSU
//         DSU s(V);
//         int ans = 0;
//         cout << "Following are the edges in the "
//             "constructed MST"
//             << endl;
//         for (auto edge : edgelist)
//         {
//             int w = edge[0];
//             int x = edge[1];
//             int y = edge[2];
//
//             // Take this edge in MST if it does
//             // not forms a cycle
//             if (s.find(x) != s.find(y))
//             {
//                 s.unite(x, y);
//                 ans += w;
//                 cout << x << " -- " << y << " == " << w
//                     << endl;
//             }
//         }
//         cout << "Minimum Cost Spanning Tree: " << ans << endl;
//
//         // Visualize the MST
//     }
// };
//
// int main()
// {
//     // Load the input image
//     Mat image = imread("C:\\Users\\TYH\\source\\repos\\DSPC\\x64\\Debug\\lena.png", IMREAD_GRAYSCALE);
//     Graph g(image.rows * image.cols);
//     // resize(image, image, Size(100, 100));
//     // imshow("Grayscale", image);
//     // int numVertices = image.rows * image.cols;
//     // for (int y = 0; y < image.rows; ++y)
//     // {
//     //     for (int x = 0; x < image.cols; ++x)
//     //     {
//     //         g.addEdge(x, y, static_cast<float>(image.at<uchar>(y, x)));
//     //     }
//     // }
//     // g.kruskals_mst();
//
//     // // Define the graph as a vector of edges
//     // vector<Edge> edges;
//     // int numVertices = image.rows * image.cols;
//     //
//     // clock_t start_graph = clock();
//     // // Calculate edge weights based on pixel intensity differences
//     // for (int y = 0; y < image.rows; ++y)
//     // {
//     //     for (int x = 0; x < image.cols; ++x)
//     //     {
//     //         int v = y * image.cols + x;
//     //         if (x < image.cols - 1)
//     //         {
//     //             int u = y * image.cols + (x + 1);
//     //             float weight = abs(image.at<uchar>(y, x) - image.at<uchar>(y, x + 1));
//     //             edges.push_back({v, u, weight});
//     //         }
//     //         if (y < image.rows - 1)
//     //         {
//     //             int u = (y + 1) * image.cols + x;
//     //             float weight = abs(image.at<uchar>(y, x) - image.at<uchar>(y + 1, x));
//     //             edges.push_back({v, u, weight});
//     //         }
//     //     }
//     // }
//     // clock_t end_graph = clock();
//     //
//     // // Sort edges by weight in ascending order
//     // sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b)
//     // {
//     //     return a.weight < b.weight;
//     // });
//     //
//     //
//     // clock_t start_kruskal = clock();
//     // kruskal(image, edges);
//     // clock_t end_kruskal = clock();
//     // cout << "graph time: " << (double)(end_graph - start_graph) / CLOCKS_PER_SEC << endl;
//     // cout << "algorithm time" << (double)(end_kruskal - start_kruskal) / CLOCKS_PER_SEC << endl;
//
//
//     waitKey(0);
//     return 0;
// }

enum CHANNELS
{
    RED = 0,
    GREEN,
    BLUE,
    INTENSITY
};

typedef struct stImage
{
    double val[4];
} Image;


typedef struct stEdge
{
    int v1;
    int v2;
    double w;
} Edge;

// IplImage *imgSrc;
// IplImage *imgDst;
// IplImage *imgGray;

Edge* E;
int *p, *rango, *C, *COL;
int nRows;
int nColumns;
double* Int;
double thao = 10000.0;

// Image ** ConvImgToDbl(IplImage *);
// void ConvDblToImg(Image **, IplImage *, bool);
void Smooth(Image**, Image**, int);
void MST_Segmentation(Image**, int);
int compare(const void*, const void*);

void make_set(int);
void link(int, int);
int find_set(int);
void union_set(int, int);

int Min(double, double);
int Max(double, double);

Mat ImageToMat(Image** img)
{
    // Create empty Mat
    Mat mat(nRows, nColumns, CV_8UC3);
    // Copy Image to Mat
    for (int i = 0; i < nRows; i++)
    {
        for (int j = 0; j < nColumns; j++)
        {
            mat.at<Vec3b>(i, j)[0] = img[i][j].val[BLUE];
            mat.at<Vec3b>(i, j)[1] = img[i][j].val[GREEN];
            mat.at<Vec3b>(i, j)[2] = img[i][j].val[RED];
        }
    }
    return mat;
}

//Original main
int d_main()
{
    int i, j;

    Image** X;
    //Image **imgR;
    //Image **imgG;
    //Image **imgB;
    Image** imgI;

    srand(time(NULL));
    //freopen("mat4.txt", "w", stdout);

    // cvNamedWindow("MST");
    // cvNamedWindow("MST Gray");
    Mat image = imread("C:\\Users\\TYH\\source\\repos\\DSPC\\x64\\Debug\\lena.png", IMREAD_GRAYSCALE);
    // imgSrc = cvLoadImage("florencia.jpg");
    // imgDst = cvCreateImage(cvSize(imgSrc->width, imgSrc->height), imgSrc->depth, imgSrc->nChannels);
    // imgGray = cvCreateImage(cvSize(imgSrc->width, imgSrc->height), imgSrc->depth, imgSrc->nChannels);

    nRows = image.rows;
    nColumns = image.cols;
    imgI = new Image*[nRows];

    // X = ConvImgToDbl(imgSrc);
    // imgI = ConvImgToDbl(imgSrc);
    //
    // ConvDblToImg(X, imgGray, true);

    // for(i=0; i<nRows; i++)
    // {
    //     for(j=0; j<nColumns; j++)
    //     {
    //         if(j == 0)
    //             printf("%.0lf", X[i][j].val[INTENSITY]);
    //         else
    //             printf(" %.0lf", X[i][j].val[INTENSITY]);
    //     }
    //     printf("\n");
    // }


    // Smooth(X, imgI, INTENSITY);

    for (i = 0; i < nRows; i++)
    {
        for (j = 0; j < nColumns; j++)
        {
            if (j == 0)
                printf("%.2lf", imgI[i][j].val[INTENSITY]);
            else
                printf(" %.2lf", imgI[i][j].val[INTENSITY]);
        }
        printf("\n");
    }


    MST_Segmentation(imgI, INTENSITY);
    imshow("Test", ImageToMat(imgI));

    // ConvDblToImg(imgI, imgDst, false);
    //
    // cvShowImage("MST", imgDst);
    // cvSaveImage("mst_tiger_2.jpg", imgDst);
    //
    // cvShowImage("MST Gray", imgGray);
    // //cvSaveImage("mstG3.jpg", imgGray);
    //
    //
    // cvWaitKey(0);
    //
    // cvReleaseImage(&imgSrc);
    // cvReleaseImage(&imgDst);
    // cvReleaseImage(&imgGray);
    //
    // cvDestroyWindow("MST");
    // cvDestroyWindow("MST Gray");

    return 0;
}

void MST_Segmentation(Image** I, int channel)
{
    int i, j, k, r, c;
    int u, v, set1, set2, newSet;
    int C1, C2;
    int nEdges = 0;
    int r1, c1, r2, c2;
    double Int1, Int2;
    double MInt, cost;

    E = new Edge[5 * nRows * nColumns];
    p = new int[5 * nRows * nColumns];
    rango = new int[5 * nRows * nColumns];
    Int = new double[5 * nRows * nColumns];
    C = new int[5 * nRows * nColumns];
    COL = new int[5 * nRows * nColumns];

    for (i = 0; i < nRows; i++)
    {
        for (j = 0; j < nColumns; j++)
        {
            k = i * nColumns + j;

            make_set(k);

            COL[k] = -1;
            Int[k] = 0.0;
            C[k] = 1;

            for (r = i - 1; r <= i; r++)
            {
                for (c = j - 1; c <= j + 1; c++)
                {
                    if (r == i && c == j)
                        continue;

                    if (r < 0 || c < 0 || c >= nColumns || r >= nRows)
                        continue;

                    E[nEdges].v1 = i * nColumns + j;
                    E[nEdges].v2 = r * nColumns + c;
                    E[nEdges].w = fabs(I[i][j].val[channel] - I[r][c].val[channel]);

                    nEdges++;
                }
            }
        }
    }

    qsort(E, nEdges, sizeof(E[0]), compare);

    //printf("nEdges: %d\n", nEdges);

    for (i = 0; i < nEdges; i++)
    {
        u = E[i].v1;
        v = E[i].v2;
        cost = E[i].w;

        set1 = find_set(u);
        set2 = find_set(v);
        Int1 = Int[set1];
        Int2 = Int[set2];
        C1 = C[set1];
        C2 = C[set2];

        MInt = Min(Int1 + thao / C1, Int2 + thao / C2);

        if (set1 != set2 && cost < MInt)
        {
            union_set(u, v);

            newSet = find_set(u);

            Int[newSet] = Max(Int1, Int2);
            Int[newSet] = Max(Int[newSet], cost);
            C[newSet] = C1 + C2;
        }
    }

    for (i = 0; i < nRows * nColumns; i++)
    {
        r1 = i / nColumns;
        c1 = i % nColumns;

        k = find_set(i);

        if (COL[k] == -1)
        {
            COL[k] = i;

            I[r1][c1].val[RED] = rand() % 255;
            I[r1][c1].val[GREEN] = rand() % 255;
            I[r1][c1].val[BLUE] = rand() % 255;
            I[r1][c1].val[INTENSITY] = (I[r1][c1].val[RED] + I[r1][c1].val[GREEN] + I[r1][c1].val[BLUE]) / 3.0;
        }
        else
        {
            r2 = COL[k] / nColumns;
            c2 = COL[k] % nColumns;

            I[r1][c1].val[RED] = I[r2][c2].val[RED];
            I[r1][c1].val[GREEN] = I[r2][c2].val[GREEN];
            I[r1][c1].val[BLUE] = I[r2][c2].val[BLUE];
            I[r1][c1].val[INTENSITY] = I[r2][c2].val[INTENSITY];
        }
    }
}

void Smooth(Image** S, Image** D, int channel)
{
    int i, j;
    int r, c;
    double sum, k;

    double M[5][5] = {
        {2, 4, 5, 4, 2},
        {4, 9, 12, 9, 4},
        {5, 12, 15, 12, 5},
        {4, 9, 12, 9, 4},
        {2, 4, 5, 4, 2}
    };

    for (i = 0; i < nRows; i++)
    {
        for (j = 0; j < nColumns; j++)
        {
            k = 0.0;
            sum = 0.0;

            for (r = i - 2; r <= i + 2; r++)
            {
                for (c = j - 2; c <= j + 2; c++)
                {
                    if (r < 0 || r >= nRows || c < 0 || c >= nColumns)
                        continue;

                    k += S[r][c].val[channel] * M[r - i + 2][c - j + 2];
                    sum += M[r - i + 2][c - j + 2];
                }
            }

            D[i][j].val[channel] = k / sum;
        }
    }
}

// Image ** ConvImgToDbl(IplImage *img)
// {
//     CvScalar color;
//     Image **T;
//     int i, j;
//     
//     T = new Image *[img->height];
//     for(i=0; i<img->height; i++)
//     {
//         T[i] = new Image[img->width];
//         for(j=0; j<img->width; j++)
//         {
//             color = cvGet2D(img, i, j);
//             
//             T[i][j].val[RED] = color.val[RED];
//             T[i][j].val[GREEN] = color.val[GREEN];
//             T[i][j].val[BLUE] = color.val[2];
//             T[i][j].val[INTENSITY] = (color.val[RED] + color.val[GREEN] + color.val[BLUE])/3.0;
//         }
//     }
//     
//     return T;
// }

// void ConvDblToImg(Image **I, IplImage *img, bool grayScale)
// {
//     int i, j;
//     int r, g, b;
//     
//     for(i=0; i<nRows; i++)
//     {
//         for(j=0; j<nColumns; j++)
//         {
//             if(grayScale == true)
//                 r = g = b = (int)I[i][j].val[INTENSITY];
//             else 
//             {
//                 r = (int)I[i][j].val[RED];
//                 g = (int)I[i][j].val[GREEN];
//                 b = (int)I[i][j].val[BLUE];
//             }
//             
//             cvSet2D(img, i, j, cvScalar(r, g, b));
//         }
//     }
// }

void make_set(int x)
{
    p[x] = x;
    rango[x] = 0;
}

void link(int x, int y)
{
    if (rango[x] > rango[y])
        p[y] = x;
    else
    {
        p[x] = y;
        if (rango[x] == rango[y])
            rango[y] = rango[y] + 1;
    }
}

int find_set(int x)
{
    if (x != p[x])
        p[x] = find_set(p[x]);
    return p[x];
}

void union_set(int x, int y)
{
    link(find_set(x), find_set(y));
}

int Min(double a, double b)
{
    if (a < b)
        return a;
    else
        return b;
}

int Max(double a, double b)
{
    if (a > b)
        return a;
    else
        return b;
}

int compare(const void* a, const void* b)
{
    Edge* sp1 = (Edge*)a;
    Edge* sp2 = (Edge*)b;

    if (sp1->w < sp2->w)
        return -1;
    else if (sp1->w > sp2->w)
        return 1;

    return 0;
}
