using System;
using System.Collections.Generic;
using System.Linq;

namespace CompressionApp
{
    public static class Huffman
    {
        private class Node
        {
            public int Symbol;
            public int Frequency;
            public Node? Left;
            public Node? Right;
            public bool IsLeaf => Left == null && Right == null;
        }

        public static (Dictionary<int, string> Table, string EncodedBits) Encode(int[] data)
        {
            // 1. Count frequencies
            var freq = data.GroupBy(v => v).ToDictionary(g => g.Key, g => g.Count());

            // 2. Build priority queue (sorted by frequency)
            var nodes = new List<Node>(freq.Select(kv => new Node { Symbol = kv.Key, Frequency = kv.Value }));

            while (nodes.Count > 1)
            {
                var ordered = nodes.OrderBy(n => n.Frequency).ToList();
                var left = ordered[0];
                var right = ordered[1];
                nodes.Remove(left);
                nodes.Remove(right);

                nodes.Add(new Node
                {
                    Frequency = left.Frequency + right.Frequency,
                    Left = left,
                    Right = right
                });
            }

            var root = nodes[0];
            var table = new Dictionary<int, string>();
            BuildCodeTable(root, "", table);

            // 3. Encode data into bitstring
            var encoded = string.Join("", data.Select(d => table[d]));
            return (table, encoded);
        }

        private static void BuildCodeTable(Node node, string prefix, Dictionary<int, string> table)
        {
            if (node.IsLeaf)
            {
                table[node.Symbol] = prefix.Length > 0 ? prefix : "0"; // edge case
                return;
            }
            BuildCodeTable(node.Left!, prefix + "0", table);
            BuildCodeTable(node.Right!, prefix + "1", table);
        }

        public static int[] Decode(string bits, Dictionary<int, string> table)
        {
            var rev = table.ToDictionary(kv => kv.Value, kv => kv.Key);
            var buffer = "";
            var result = new List<int>();

            foreach (char b in bits)
            {
                buffer += b;
                if (rev.TryGetValue(buffer, out int symbol))
                {
                    result.Add(symbol);
                    buffer = "";
                }
            }

            return result.ToArray();
        }
    }
}
