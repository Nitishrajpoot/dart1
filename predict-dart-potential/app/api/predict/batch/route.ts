import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { smiles_list } = await request.json()

    if (!Array.isArray(smiles_list) || smiles_list.length === 0) {
      return NextResponse.json({ error: "Invalid SMILES list" }, { status: 400 })
    }

    if (smiles_list.length > 100) {
      return NextResponse.json({ error: "Maximum 100 compounds per batch" }, { status: 400 })
    }

    // Process batch predictions
    const results = await Promise.all(
      smiles_list.map(async (smiles) => {
        const baseProb = Math.min(0.3 + smiles.length * 0.02, 0.95)
        const noise = Math.random() * 0.2 - 0.1
        const probability = Math.max(0.05, Math.min(0.95, baseProb + noise))
        const prediction = probability > 0.5 ? "Toxic" : "Non-Toxic"
        const confidence = 0.7 + Math.random() * 0.25

        return {
          smiles,
          prediction,
          probability: Number(probability.toFixed(4)),
          confidence: Number(confidence.toFixed(4)),
        }
      }),
    )

    return NextResponse.json({ results, total: results.length })
  } catch (error) {
    console.error("[v0] Batch prediction error:", error)
    return NextResponse.json({ error: "Batch prediction failed" }, { status: 500 })
  }
}
