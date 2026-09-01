#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <string>

using std::string;

class RuntimeException {
 public:
  RuntimeException(const string& err) : errMsg(err) {}
  string getMessage() const { return errMsg; }

 private:
  string errMsg;
};

class IndexOutOfBounds : public RuntimeException {
 public:
  IndexOutOfBounds(const string& err) : RuntimeException(err) {}
};

class MatrixSizeIncompatible : public RuntimeException {
 public:
  MatrixSizeIncompatible(const string& err) : RuntimeException(err) {}
};

#endif
