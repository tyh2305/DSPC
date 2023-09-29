// #include <iostream>
// #include <vector>
// #include <opencv2/highgui.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/core/mat.hpp>
// #include <omp.h>
// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>
// #include <cooperative_groups.h>
// using namespace cv;
// using namespace std;
//
// // To convert an image to graph to be used by MST
//
// class Color
// {
// public:
//     int R;
//     int G;
//     int B;
//
//     Color(int R, int G, int B)
//     {
//         this->R = R;
//         this->G = G;
//         this->B = B;
//     }
//
//     Color(): R(255), G(255), B(255)
//     {
//     }
//
//     Vec3b toVec3b()
//     {
//         return Vec3b(B, G, R);
//     }
// };
//
// class Pixel
// {
// public:
//     int x;
//     int y;
//     int intensity;
//
//     Pixel()
//     {
//         x = 0;
//         y = 0;
//         intensity = 0;
//     };
//
//     Pixel(int x, int y, int intensity): x(x), y(y), intensity(intensity)
//     {
//     }
//
//     bool operator==(const Pixel& other)
//     {
//         return x == other.x && y == other.y;
//     }
//
//     friend ostream& operator<<(ostream& os, const Pixel& p)
//     {
//         os << "(" << p.x << ", " << p.y << ")";
//         return os;
//     }
// };
//
// class Edge
// {
// public:
//     Pixel src;
//     Pixel des;
//     int weight;
//
//     Edge()
//     {
//         src = Pixel();
//         des = Pixel();
//         weight = 0;
//     };
//
//     // friend ostream& operator<<(ostream& os, const Pixel& p)
//     // {
//     //     os << "(" << p.x << ", " << p.y << ")";
//     //     return os;
//     // }
// };
//
// struct Region
// {
//     vector<Pixel> points;
//     Color color;
// };
//
// class DisjointSet
// {
// public:
//     DisjointSet(int size): parent(size), rank(size, 0)
//     {
//         for (int i = 0; i < size; ++i)
//         {
//             parent[i] = i;
//         }
//     }
//
//     int find(int v)
//     {
//         if (v == parent[v])
//         {
//             return v;
//         }
//         return parent[v] = find(parent[v]);
//     }
//
//     void unionSets(int a, int b)
//     {
//         a = find(a);
//         b = find(b);
//
//         if (a != b)
//         {
//             if (rank[a] < rank[b])
//             {
//                 swap(a, b);
//             }
//             parent[b] = a;
//             if (rank[a] == rank[b])
//             {
//                 rank[a]++;
//             }
//         }
//     }
//
// private:
//     vector<int> parent;
//     vector<int> rank;
// };
//
// bool compareEdges(const Edge& a, const Edge& b)
// {
//     return a.weight < b.weight;
// }
//
// int PixelToInt(Pixel pix, int col)
// {
//     return pix.y * col + pix.x;
// }
//
// vector<Point> pixelToPoint(vector<Pixel> pixels)
// {
//     vector<Point> points;
//     for (const Pixel& pixel : pixels)
//     {
//         points.push_back(Point(pixel.x, pixel.y));
//     }
//     return points;
// }
//
// vector<Pixel> unionPixel(vector<Pixel> a, vector<Pixel> b)
// {
//     vector<Pixel> result;
//     result.insert(result.end(), a.begin(), a.end());
//     result.insert(result.end(), b.begin(), b.end());
//     return result;
// }
//
// vector<Edge> kruskalMST(vector<Edge>& edges, int col, int row)
// {
//     vector<Edge> MinimumSpanningTree;
//     sort(edges.begin(), edges.end(), compareEdges);
//
//     DisjointSet ds(col * row);
//
//     for (const Edge& edge : edges)
//     {
//         if (ds.find(PixelToInt(edge.src, col)) != ds.find(PixelToInt(edge.des, col)))
//         {
//             MinimumSpanningTree.push_back(edge);
//             ds.unionSets(PixelToInt(edge.src, col), PixelToInt(edge.des, col));
//         }
//     }
//
//     return MinimumSpanningTree;
// }
//
// vector<Pixel> convertMatToPixel(Mat image)
// {
//     vector<Pixel> pixels;
//     for (int i = 0; i < image.rows; i++)
//     {
//         for (int j = 0; j < image.cols; j++)
//         {
//             Pixel p;
//             p.x = i;
//             p.y = j;
//             Color color;
//             p.intensity = image.at<uchar>(i, j);
//             pixels.push_back(p);
//         }
//     }
//     return pixels;
// }
//
// vector<Edge> convertPixelToEdge(vector<Pixel> pixels, int row, int col)
// {
//     vector<Edge> edges;
//     for (int y = 0; y < row; y++)
//     {
//         for (int x = 0; x < col; x++)
//         {
//             // Pixel 1
//             Pixel src = pixels[y * col + x];
//             if (x < col - 1)
//             {
//                 // Last element in row
//                 Pixel des = pixels[y * col + x + 1];
//                 Edge e;
//                 e.src = src;
//                 e.des = des;
//                 e.weight = abs(src.intensity - des.intensity);
//                 edges.push_back(e);
//             }
//             if (y < row - 1)
//             {
//                 // Last row
//                 Pixel px2 = pixels[(y + 1) * col + x];
//                 Edge e;
//                 e.src = src;
//                 e.des = px2;
//                 e.weight = abs(src.intensity - px2.intensity);
//                 edges.push_back(e);
//             }
//         }
//     }
//     return edges;
// }
//
// vector<Edge> processMST(vector<Edge> mst)
// {
//     // Calculate average and standard deviation of the weights
//     cout << "Ori MST size: " << mst.size() << endl;
//     int sum = 0;
//     for (const Edge& edge : mst)
//     {
//         sum += edge.weight;
//     }
//     int average = sum / mst.size();
//     int standardDeviation = 0;
//     for (const Edge& edge : mst)
//     {
//         standardDeviation += pow(edge.weight - average, 2);
//     }
//     standardDeviation = sqrt(standardDeviation / mst.size());
//     cout << "Avg: " << average << endl;
//     cout << "Std: " << standardDeviation << endl;
//     // Remove outliers
//     vector<Edge> newMST;
//     for (const Edge& edge : mst)
//     {
//         if (edge.weight < average + standardDeviation)
//         {
//             newMST.push_back(edge);
//         }
//     }
//     cout << "New MST size: " << newMST.size() << endl;
//     sort(newMST.begin(), newMST.end(), compareEdges);
//     return newMST;
// }
//
// Mat visualizeMST(const vector<Edge>& mst, Mat oriImage)
// {
//     Mat outputImage = oriImage.clone(); // Clone the original image to prevent modification
//
//     for (const Edge& edge : mst)
//     {
//         Scalar color(rand() & 255, rand() & 255, rand() & 255); // Generate a random color
//         vector<Point> points;
//
//         // Create a polygon using the source and destination vertices of the MST edge
//         points.push_back(Point(edge.src.x, edge.src.y));
//         points.push_back(Point(edge.des.x, edge.des.y));
//
//         // Draw the polygon on the output image
//         const Point* pts = (const Point*)Mat(points).data;
//         int npts = Mat(points).rows;
//         fillPoly(outputImage, &pts, &npts, 1, color);
//     }
//
//     return outputImage;
// }
//
// vector<Edge> convertMatToEdge(Mat image)
// {
//     int row = image.rows;
//     int col = image.cols;
//     return convertPixelToEdge(convertMatToPixel(image), row, col);
// }
//
// vector<Region> segmentOpenMP(const vector<Edge>& mst)
// {
//     // Mat labels(oriImage.rows, oriImage.cols, CV_32S, Scalar(0)); // Labels for regions
//     // vector<Pixel> labels;
//     vector<Region> regions;
//     int thread = 16;
//     omp_set_num_threads(thread);
//
//     int currentLabel = 1; // Start labeling from 1 (0 is reserved for background)
//
//     // Start time with OpenMP
//     double startTime = omp_get_wtime();
// #pragma omp parallel for shared(regions, currentLabel)
//     for (int i = 0; i < mst.size(); i++)
//     {
//         // cout << "Current progress: " << currentLabel << "/" << mst.size() << endl;
//         currentLabel++;
// #pragma omp critical
//         {
//             const Edge& edge = mst[i];
//
//             // find is there a region that contains the source pixel
//             bool srcFound = false;
//             bool desFound = false;
//             int srcRegionIndex = -1;
//             int desRegionIndex = -1;
//             if (regions.size() == 0)
//             {
//                 srcFound = false;
//                 desFound = false;
//             }
//             else
//             {
//                 for (int i = 0; i < regions.size(); i++)
//                 {
//                     for (int j = 0; j < regions[i].points.size(); j++)
//                     {
//                         const Pixel& pixel = regions[i].points[j];
//                         if (pixel.x == edge.src.x && pixel.y == edge.src.y)
//                         {
//                             srcFound = true;
//                             if (srcRegionIndex == -1) srcRegionIndex = i;
//                         }
//                         if (pixel.x == edge.des.x && pixel.y == edge.des.y)
//                         {
//                             desFound = true;
//                             if (desRegionIndex == -1) desRegionIndex = i;
//                         }
//                         if (desFound || srcFound) break;
//                     }
//                 }
//             }
//
//             // If both source and destination pixels are not in any region, create a new region
//             if (!srcFound && !desFound)
//             {
//                 // Create a new region
//                 Region region;
//                 region.points.push_back(edge.src);
//                 region.points.push_back(edge.des);
//                 region.color = Color(rand() & 255, rand() & 255, rand() & 255);
//                 regions.push_back(region);
//             }
//             else if (srcFound && !desFound)
//             {
//                 // Add destination pixel to the region
//                 regions[srcRegionIndex].points.push_back(edge.des);
//             }
//             else if (desFound && !srcFound)
//             {
//                 // Add source pixel to the region
//                 regions[desRegionIndex].points.push_back(edge.src);
//             }
//             else
//             {
//                 // Both are found in the regions
//                 // Merge the regions
//                 regions[srcRegionIndex].points = unionPixel(regions[srcRegionIndex].points,
//                                                             regions[desRegionIndex].points);
//                 // Remove the destination region
//                 regions.erase(regions.begin() + desRegionIndex);
//             }
//         }
//     }
//
//     // End time with OpenMP
//     double endTime = omp_get_wtime();
//     cout << "Time elapsed for openMP: " << endTime - startTime << endl;
//     return regions;
// }
//
//
// vector<Region> segmentRegions(const vector<Edge>& mst)
// {
//     // Mat labels(oriImage.rows, oriImage.cols, CV_32S, Scalar(0)); // Labels for regions
//     // vector<Pixel> labels;
//     vector<Region> regions;
//
//     int currentLabel = 1; // Start labeling from 1 (0 is reserved for background)
//
//     // Start time with OpenMP
//     double startTime = omp_get_wtime();
//     for (const Edge& edge : mst)
//     {
//         // cout << "Current progress: " << currentLabel << "/" << mst.size() << endl;
//         currentLabel++;
//         // find is there a region that contains the source pixel
//         bool srcFound = false;
//         bool desFound = false;
//         int srcRegionIndex = -1;
//         int desRegionIndex = -1;
//
//         for (int i = 0; i < regions.size(); i++)
//         {
//             // Algorithm to find if a pixel is in a region
//             for (Pixel& pixel : regions[i].points)
//             {
//                 if (pixel.x == edge.src.x && pixel.y == edge.src.y)
//                 {
//                     srcFound = true;
//                     if (srcRegionIndex == -1) srcRegionIndex = i;
//                 }
//                 if (pixel.x == edge.des.x && pixel.y == edge.des.y)
//                 {
//                     desFound = true;
//                     if (desRegionIndex == -1) desRegionIndex = i;
//                 }
//             }
//         }
//
//         // If both source and destination pixels are not in any region, create a new region
//         if (!srcFound && !desFound)
//         {
//             // Create a new region
//             Region region;
//             region.points.push_back(edge.src);
//             region.points.push_back(edge.des);
//             region.color = Color(rand() & 255, rand() & 255, rand() & 255);
//             regions.push_back(region);
//         }
//         else if (srcFound && !desFound)
//         {
//             // Add destination pixel to the region
//             regions[srcRegionIndex].points.push_back(edge.des);
//         }
//         else if (desFound && !srcFound)
//         {
//             // Add source pixel to the region
//             regions[desRegionIndex].points.push_back(edge.src);
//         }
//         else
//         {
//             // Both are found in the regions
//             // Merge the regions
//             regions[srcRegionIndex].points = unionPixel(regions[srcRegionIndex].points,
//                                                         regions[desRegionIndex].points);
//             // Remove the destination region
//             regions.erase(regions.begin() + desRegionIndex);
//         }
//     }
//     // End time with OpenMP
//     double endTime = omp_get_wtime();
//     cout << "Time elapsed for default: " << endTime - startTime << endl;
//     return regions;
// }
//
// Mat drawRegions(Mat oriImage, vector<Region> regions)
// {
//     // Create a blank output image
//     Mat outputImage(oriImage.size(), oriImage.type(), Scalar(255, 255, 255));
//
//     // Draw regions on the output image
//     for (Region& region : regions)
//     {
//         // Draw region if the point x or y axis changes are more than 2
//         int minX = region.points[0].x;
//         int maxX = region.points[0].x;
//         int minY = region.points[0].y;
//         int maxY = region.points[0].y;
//         for (Pixel p : region.points)
//         {
//             if (p.x < minX) minX = p.x;
//             if (p.x > maxX) maxX = p.x;
//             if (p.y < minY) minY = p.y;
//             if (p.y > maxY) maxY = p.y;
//         }
//
//         // Calculate delta
//         int deltaX = maxX - minX;
//         int deltaY = maxY - minY;
//         if (deltaY > 0)
//         {
//             polylines(outputImage, pixelToPoint(region.points), false, Vec3b(255, 0, 0), 1, LINE_AA, 0);
//         }
//     }
//
//     return outputImage;
// }
//
// __global__ void processRegions(Edge* mst, Region* regions, int mstSize)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     int regionSize = 0;
//     if (tid < mstSize)
//     {
//         Edge edge = mst[tid];
//
//         // Find if there is a region that contains the source pixel
//         bool srcFound = false;
//         bool desFound = false;
//         int srcRegionIndex = -1;
//         int desRegionIndex = -1;
//         int index = 0;
//
//         for (int i = 0; i < regionSize; i++)
//         {
//             Region region = regions[i];
//             index++;
//             for (Pixel& pixel : region.points)
//             {
//                 if (pixel.x == edge.src.x && pixel.y == edge.src.y)
//                 {
//                     srcFound = true;
//                     if (srcRegionIndex == -1) srcRegionIndex = index;
//                 }
//                 if (pixel.x == edge.des.x && pixel.y == edge.des.y)
//                 {
//                     desFound = true;
//                     if (desRegionIndex == -1) desRegionIndex = index;
//                 }
//             }
//         }
//
//         // If both source and destination pixels are not in any region, create a new region
//         if (!srcFound && !desFound)
//         {
//             // Create a new region
//             Region newRegion;
//             newRegion.points.push_back(edge.src);
//             newRegion.points.push_back(edge.des);
//             newRegion.color = Color(rand() & 255, rand() & 255, rand() & 255);
//             regions[regionSize] = newRegion;
//             regionSize++;
//         }
//         else if (srcFound && !desFound)
//         {
//             // Add destination pixel to the source region
//             regions[srcRegionIndex].points.push_back(edge.des);
//         }
//         else if (desFound && !srcFound)
//         {
//             // Add source pixel to the destination region
//             regions[desRegionIndex].points.push_back(edge.src);
//         }
//         else
//         {
//             // Both are found in different regions
//             // Merge the regions
//             regions[srcRegionIndex].points.insert(
//                 regions[srcRegionIndex].points.end(),
//                 regions[desRegionIndex].points.begin(),
//                 regions[desRegionIndex].points.end()
//             );
//         }
//     }
// }
//
// int cmain()
// {
//     Mat image = imread("C:\\Users\\TYH\\source\\repos\\DSPC\\x64\\Debug\\lena.png");
//     cout << "Image size: " << image.cols << "x" << image.rows << endl;
//
//     // Downscale image by 10
//     // resize(image, image, Size(image.cols / 10, image.rows / 10));
//     cout << "New Image size: " << image.cols << "x" << image.rows << endl;
//
//     cout << "Start converting image to graph" << endl;
//     vector<Edge> edges = convertMatToEdge(image);
//     cout << "Finish converting image to graph" << endl;
//
//     cout << "Start MST" << endl;
//     vector<Edge> mst = kruskalMST(edges, image.cols, image.rows);
//     mst = processMST(mst);
//     cout << "Finish MST" << endl;
//
//     cout << "Start visualizing MST" << endl;
//
//     // To parallelize
//     // For CUDA use
//     // Allocate memory on GPU
//     Edge* d_mst;
//     Region* d_regions;
//     cudaMalloc(&d_mst, mst.size() * sizeof(Edge));
//     cudaMalloc(&d_regions, mst.size() * sizeof(Region));
//     // Copy data from CPU to GPU
//     cudaMemcpy(d_mst, mst.data(), mst.size() * sizeof(Edge), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_regions, d_regions, mst.size() * sizeof(Region), cudaMemcpyHostToDevice);
//     // Define grid and block dimensions
//     int blockSize = 256;
//     int gridSize = (mst.size() + blockSize - 1) / blockSize;
//     processRegions(d_mst, d_regions, mst.size());
//     cudaFree(d_mst);
//     cudaFree(d_regions);
//     // const vector<Region> regions = segmentRegions(mst);
//     vector<Region> regions;
//     int regionSize = sizeof(d_regions) / sizeof(Region);
//     // Copy data from GPU to CPU
//     for (int i = 0; i < regionSize; i++)
//     {
//         regions.push_back(d_regions[i]);
//     }
//     const Mat outputImage = drawRegions(image, regions);
//
//     // Upscale by 10 times
//     // resize(outputImage, outputImage, Size(outputImage.cols * 10, outputImage.rows * 10));
//     imshow("MST", outputImage);
//     cout << "Finish visualizing MST" << endl;
//
//     waitKey(0);
//     return 0;
// }
