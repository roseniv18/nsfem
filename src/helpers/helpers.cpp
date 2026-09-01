#include "helpers/helpers.h"

std::string strip_quotes(std::string& str) {
  if (str.size() >= 2 && str.front() == '"' && str.back() == '"') {
    return str.substr(1, str.size() - 2);
  }

  return str;
}
