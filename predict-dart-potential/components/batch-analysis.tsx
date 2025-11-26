"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { AlertCircle, Loader2, Download } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface BatchResult {
  smiles: string
  prediction: string
  probability: number
  confidence: number
}

export function BatchAnalysis() {
  const [smilesInput, setSmilesInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [results, setResults] = useState<BatchResult[]>([])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError("")
    setResults([])

    const smilesList = smilesInput.split("\n").filter((s) => s.trim())

    try {
      const response = await fetch("/api/predict/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles_list: smilesList }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Batch prediction failed")
      }

      const data = await response.json()
      setResults(data.results)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setLoading(false)
    }
  }

  const downloadResults = () => {
    const csv = [
      "SMILES,Prediction,Probability,Confidence",
      ...results.map((r) => `"${r.smiles}",${r.prediction},${r.probability.toFixed(4)},${r.confidence.toFixed(4)}`),
    ].join("\n")

    const blob = new Blob([csv], { type: "text/csv" })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `dart-predictions-${new Date().toISOString()}.csv`
    a.click()
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Batch Chemical Analysis</CardTitle>
          <CardDescription>
            Enter multiple SMILES strings (one per line) to analyze multiple compounds simultaneously
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="batch-smiles">SMILES List</Label>
              <Textarea
                id="batch-smiles"
                placeholder={"CC(=O)OC1=CC=CC=C1C(=O)O\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C\nCCO"}
                value={smilesInput}
                onChange={(e) => setSmilesInput(e.target.value)}
                className="min-h-[200px] font-mono text-sm"
                required
              />
              <p className="text-xs text-muted-foreground">Enter one SMILES string per line (maximum 100 compounds)</p>
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <Button type="submit" className="w-full" disabled={loading || !smilesInput}>
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing Batch...
                </>
              ) : (
                "Analyze Batch"
              )}
            </Button>
          </form>
        </CardContent>
      </Card>

      {results.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Results ({results.length} compounds)</CardTitle>
                <CardDescription>Batch analysis complete</CardDescription>
              </div>
              <Button onClick={downloadResults} variant="outline" size="sm">
                <Download className="mr-2 h-4 w-4" />
                Export CSV
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {results.map((result, idx) => (
                <div key={idx} className="flex items-center justify-between rounded-lg border bg-card p-4">
                  <div className="flex-1 space-y-1">
                    <p className="break-all font-mono text-sm">{result.smiles}</p>
                    <p className="text-xs text-muted-foreground">
                      Probability: {Math.round(result.probability * 100)}% | Confidence:{" "}
                      {Math.round(result.confidence * 100)}%
                    </p>
                  </div>
                  <Badge
                    variant={result.prediction === "Toxic" ? "destructive" : "default"}
                    className={result.prediction !== "Toxic" ? "bg-accent text-accent-foreground" : ""}
                  >
                    {result.prediction}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
