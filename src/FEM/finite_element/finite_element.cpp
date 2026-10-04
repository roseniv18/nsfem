#include "finite_element.h"
#include "P1_element.h"

std::unique_ptr<FE> FE::build_fe_type(const FEType& type) {
  switch (type) {
    case FEType::P1:
      return std::make_unique<P1_FE>();
    default:
      throw std::invalid_argument("Unknown FE Type");
  }
}
