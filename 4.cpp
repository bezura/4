#include <iostream>
#include <vector>

using namespace std;

class Matrix;
class Vector;

Vector operator*(const Matrix& matrix, const Vector& vector);
Vector operator*(const Vector& vector, const Matrix& matrix);
Vector solveLinearSystem(const Matrix& A, const Vector& b);

class Matrix {

protected:
    int rows;
    int cols;
    vector<vector<double>> data;

public:
    Matrix(int numRows, int numCols) : rows(numRows), cols(numCols) {
        data.resize(rows, vector<double>(cols));
    }

    void input() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cin >> data[i][j];
            }
        }
    }

    void print() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cout << data[i][j] << " ";
            }
            cout << endl;
        }
    }

    Matrix operator+(const Matrix& other) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    Matrix operator-(const Matrix& other) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    Matrix operator*(const Matrix& other) const {
        int otherCols = other.cols;
        Matrix result(rows, otherCols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < otherCols; j++) {
                for (int k = 0; k < cols; k++) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }

    Matrix operator/(double scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] / scalar;
            }
        }
        return result;
    }

    friend Vector operator*(const Matrix& matrix, const Vector& vector);
    friend Vector operator*(const Vector& vector, const Matrix& matrix);
    friend Vector solveLinearSystem(const Matrix& A, const Vector& b);
};

class Vector {

protected:
    int n;
    vector<double> v;
public:

    Vector(int size) {
        n = size;
        v.resize(n);
    }
    void input() {
        for (int i = 0; i < this->n; i++) {
            cin >> v[i];
        }
    }
    void print() {
        for (int i = 0; i < n; i++) {
            cout << v[i] << " ";
        }
        cout << endl;
    }
    Vector operator+ (const Vector& other) const {
        Vector res(n);
        for (int i = 0; i < n; i++) {
            res.v[i] = v[i] + other.v[i];
        }
        return res;
    }
    Vector operator- (const Vector& other) const {
        Vector res(n);
        for (int i = 0; i < n; i++) {
            res.v[i] = v[i] - other.v[i];
        }
        return res;
    }
    double operator* (const Vector& other) const {
        double res = 0;
        for (int i = 0; i < n; i++) {
            res += v[i] * other.v[i];
        }
        return res;
    }
    Vector operator* (double scalar) const {
        Vector res(n);
        for (int i = 0; i < n; i++) {
            res.v[i] = v[i] * scalar;
        }
        return res;
    }

    Vector operator/ (double scalar) const {
        Vector res(n);
        for (int i = 0; i < n; i++) {
            res.v[i] = v[i] / scalar;
        }
        return res;
    }

    friend Vector operator*(const Matrix& matrix, const Vector& vector);
    friend Vector operator*(const Vector& vector, const Matrix& matrix);
    friend Vector solveLinearSystem(const Matrix& A, const Vector& b);
};

Vector operator*(const Matrix& matrix, const Vector& vector) {
    if (matrix.cols != vector.n) {
        throw runtime_error("Ошибка: размерности матрицы и вектора не совпадают для умножения");
    }

    Vector result(matrix.rows);

    for (int i = 0; i < matrix.rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < matrix.cols; j++) {
            sum += matrix.data[i][j] * vector.v[j];
        }
        result.v[i] = sum;
    }

    return result;
}

Vector operator*(const Vector& vector, const Matrix& matrix) {
    if (matrix.rows != vector.n) {
        throw runtime_error("Ошибка: размерности вектора и матрицы не совпадают для умножения");
    }

    Vector result(matrix.cols);

    for (int j = 0; j < matrix.cols; j++) {
        double sum = 0.0;
        for (int i = 0; i < matrix.rows; i++) {
            sum += vector.v[i] * matrix.data[i][j];
        }
        result.v[j] = sum;
    }

    return result;
}

Vector solveLinearSystem(const Matrix& A, const Vector& b) {
    int n = A.rows;
    int m = A.cols;

    if (n != m) {
        throw runtime_error("Ошибка: матрица A должна быть квадратной");
    }

    if (n != b.n) {
        throw runtime_error("Ошибка: размерность вектора b не соответствует матрице A");
    }

    Matrix augmentedMatrix(n, n + 1);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            augmentedMatrix.data[i][j] = A.data[i][j];
        }
        augmentedMatrix.data[i][n] = b.v[i];
    }

    // Прямой ход метода Гаусса
    for (int i = 0; i < n - 1; i++) {
        int maxRow = i;
        double maxValue = augmentedMatrix.data[i][i];
        for (int k = i + 1; k < n; k++) {
            if (abs(augmentedMatrix.data[k][i]) > abs(maxValue)) {
                maxRow = k;
                maxValue = augmentedMatrix.data[k][i];
            }
        }

        if (maxRow != i) {
            swap(augmentedMatrix.data[i], augmentedMatrix.data[maxRow]);
        }

        for (int k = i + 1; k < n; k++) {
            double factor = augmentedMatrix.data[k][i] / augmentedMatrix.data[i][i];
            for (int j = i; j < n + 1; j++) {
                augmentedMatrix.data[k][j] -= factor * augmentedMatrix.data[i][j];
            }
        }
    }

    Vector x(n);
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++) {
            sum += augmentedMatrix.data[i][j] * x.v[j];
        }
        x.v[i] = (augmentedMatrix.data[i][n] - sum) / augmentedMatrix.data[i][i];
    }

    return x;
}

int main() {
    int numRows, numCols;
    cout << "Enter the dimensions of the matrix: ";
    cin >> numRows >> numCols;

    Matrix matrix(numRows, numCols);
    cout << "Enter the elements of the matrix: " << endl;
    matrix.input();

    Vector vector(numCols);
    cout << "Enter the elements of the vector: ";
    vector.input();

    Vector result1 = matrix * vector;
    cout << "Result of matrix-vector multiplication: ";
    result1.print();

    Vector result2 = vector * matrix;
    cout << "Result of vector-matrix multiplication: ";
    result2.print();

    Vector b(numRows);
    cout << "Enter the elements of the vector b for the linear system: ";
    b.input();

    Vector solution = solveLinearSystem(matrix, b);
    cout << "Solution to the linear system Ax = b: ";
    solution.print();

    return 0;
}
