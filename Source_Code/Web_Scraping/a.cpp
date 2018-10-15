/**
 * This file is used to output text that is not within the <story> and </story> tags
 */
#include <bits/stdc++.h>
using namespace std;
#define W(X) while (X)
#define fi first
#define se second
#define A(X) begin(X), end(X)
#define COUTM(X)                                                               \
  for (const auto &i : X) {                                                    \
    cerr << i.fi << " " << i.se << "\n";                                       \
  }
#define COUT(X)                                                                \
  for (const auto &i : X) {                                                    \
    cerr << i << "\n";                                                         \
  }
#define FA(i, v) for (auto &i : v)
#define V(X) cerr << #X << "->" << X << "\n";
#define VP(X)                                                                  \
  cerr << #X.first << "->" << X.first << #X.second << "->" << X.second << "\n";
#define S(X) cerr << #X << "->" << X << " ";
#define P(X) cerr << X << "\n";
#define PS(X) cerr << X << " ";
#define NL cout << "\n";
using vi = vector<int>;
using pii = pair<int, int>;
using ll = long long;
using li = long int;
#define F(X, Y, Z) for (ll X = Y; X < Z; X++)
#define R(X, Y, Z) for (ll X = Y; X > Z; X--)
#define IS_EVEN(X) ((X & 1) == 0)
#define BMT(mask, i) ((mask & (1 << i)) != 0)
#define BMS(mask, i) (mask |= (1 << i))
#define BMU(mask, i) (mask &= ~(1 << i))
#define umap unordered_map
int main() {
  int cnt = 0;
  string s;
  while (cin >> s) {
    //cout << "Input string = " << s << "\n";
    if (s == "<story>") {
      //cout << "Matched story begin";
      cnt++;
    } else if (s == "</story>" || s == "</story>.") {
      //cout << "Matched story end";
      cnt--;
    } else if (cnt == 0) {
      cout << s << " ";
    }
  }
  return 0;
}
