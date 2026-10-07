#include <gtest/gtest.h>

#include "FEM/finite_element/P2_element.h"
#include "helpers/helpers.h"

TEST(P2_Element_Test, NodalProperty) {
  const P2_FE fe;
  const auto ref_nodes = fe.get_ref_nodes();

  for (int i = 0; i < fe.get_ndofs(); i++) {
    auto bfs = fe.evaluate_bfs(ref_nodes.at(i).xi, ref_nodes.at(i).eta);

    for (int j = 0; j < 3; j++) {
      double expected{};
      if (i == j)
        expected = 1.0;
      else
        expected = 0.0;

      EXPECT_NEAR(bfs[j], expected, test_dtol);
    }
  }
}
