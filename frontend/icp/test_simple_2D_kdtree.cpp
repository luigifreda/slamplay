#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

struct Point {
    int x;
    int y;
};

struct Node {
    Point point;
    Node* left;
    Node* right;

    Node(Point p) : point(p), left(nullptr), right(nullptr) {}
};

// Function to compare points based on the current level
bool comparePoints(const Point& p1, const Point& p2, int level) 
{
    if (level % 2 == 0) {
        return p1.x < p2.x;
    } else {
        return p1.y < p2.y;
    }
}

// Function to build a KD-tree
Node* buildKDTree(const std::vector<Point>& points, int level) 
{
    if (points.empty()) {
        return nullptr;
    }

    // Sort points based on the current level
    std::vector<Point> sortedPoints = points;
    std::sort(sortedPoints.begin(), sortedPoints.end(),
              [level](const Point& p1, const Point& p2) {
                  return comparePoints(p1, p2, level);
              });

    // Find the median point
    int medianIndex = sortedPoints.size() / 2;
    Point medianPoint = sortedPoints[medianIndex];

    // Create a new node with the median point
    Node* node = new Node(medianPoint);

    // Recursively build the left and right subtrees
    std::vector<Point> leftPoints(sortedPoints.begin(), sortedPoints.begin() + medianIndex);
    std::vector<Point> rightPoints(sortedPoints.begin() + medianIndex + 1, sortedPoints.end());
    node->left = buildKDTree(leftPoints, level + 1);
    node->right = buildKDTree(rightPoints, level + 1);

    return node;
}

void searchRange(Node* node, const Point& lower, const Point& upper, int level, std::vector<Point>& result) 
{
    if (node == nullptr) {
        return;
    }

    if (node->point.x >= lower.x && node->point.x <= upper.x &&
        node->point.y >= lower.y && node->point.y <= upper.y) {
        result.push_back(node->point);
    }

    if (comparePoints(node->point, lower, level)) 
    {
        searchRange(node->right, lower, upper, level + 1, result);
    } 
    else if (comparePoints(upper, node->point, level)) 
    {
        searchRange(node->left, lower, upper, level + 1, result);
    } 
    else 
    {
        searchRange(node->left, lower, upper, level + 1, result);
        searchRange(node->right, lower, upper, level + 1, result);
    }
}


// Function to calculate the Euclidean distance between two points
double calculateDistance(const Point& p1, const Point& p2) {
    int dx = p1.x - p2.x;
    int dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Function to search for the nearest neighbor in the KD-tree
void searchNearestNeighbor(Node* node, const Point& target, int level, Point& nearest, double& minDistance) 
{
    if (node == nullptr) {
        return;
    }

    double distance = calculateDistance(node->point, target);

    if (distance < minDistance) {
        nearest = node->point;
        minDistance = distance;
    }

    if (comparePoints(node->point, target, level)) 
    {
        searchNearestNeighbor(node->right, target, level + 1, nearest, minDistance);
    } 
    else 
    {
        searchNearestNeighbor(node->left, target, level + 1, nearest, minDistance);
    }

    // Check the other subtree if necessary
    double axisDistance = (level % 2 == 0) ? std::abs(target.x - node->point.x) : std::abs(target.y - node->point.y);
    if (axisDistance < minDistance)
    {
        if (comparePoints(node->point, target, level)) 
        {
            searchNearestNeighbor(node->left, target, level + 1, nearest, minDistance);
        } 
        else 
        {
            searchNearestNeighbor(node->right, target, level + 1, nearest, minDistance);
        }
    }
}

// Function to print the points in the KD-tree in inorder traversal
void printInorder(Node* node) 
{
    if (node == nullptr) {
        return;
    }

    printInorder(node->left);
    std::cout << "(" << node->point.x << ", " << node->point.y << ") ";
    printInorder(node->right);
}

// a very simple implementation of a 2D KD-tree
int main() 
{
    std::vector<Point> points = {{3, 6}, {17, 15}, {13, 15}, {6, 12}, {9, 1}, {2, 7}, {10, 19}};

    Node* root = buildKDTree(points, 0);

    std::cout << "Inorder traversal of the KD-tree:\n";
    printInorder(root);
    std::cout << "\n";

    Point lowerBound = {5, 5};
    Point upperBound = {15, 15};
    std::vector<Point> result;

    searchRange(root, lowerBound, upperBound, 0, result);

    std::cout << "Points within the range (" << lowerBound.x << ", " << lowerBound.y << ") to (" << upperBound.x << ", " << upperBound.y << "):\n";
    for (const Point& p : result) {
        std::cout << "(" << p.x << ", " << p.y << ") ";
    }
    std::cout << "\n";


    Point target = {7, 8};
    Point nearestPoint;
    double minDistance = std::numeric_limits<double>::max();

    searchNearestNeighbor(root, target, 0, nearestPoint, minDistance);

    std::cout << "Nearest neighbor to (" << target.x << ", " << target.y << "): ";
    std::cout << "(" << nearestPoint.x << ", " << nearestPoint.y << ")\n";


    return 0;
}