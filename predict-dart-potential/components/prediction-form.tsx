"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { AlertCircle, Loader2 } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { PredictionResult } from "@/components/prediction-result"

interface PredictionData {
  prediction: string
  probability: number
  confidence: number
  model_used: string
  features_used: number
  smiles: string
}

export function PredictionForm() {
  const [smiles, setSmiles] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [result, setResult] = useState<PredictionData | null>(null)

  const exampleSmiles = [
    { name: "Aspirin", smiles: "CC(=O)OC1=CC=CC=C1C(=O)O" },
    { name: "Caffeine", smiles: "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" },
    { name: "Ethanol", smiles: "CCO" },
  ]

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError("")
    setResult(null)

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Prediction failed")
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Enter Chemical Structure</CardTitle>
          <CardDescription>Input a chemical compound using SMILES notation to predict DART toxicity</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="smiles">SMILES String</Label>
              <Textarea
                id="smiles"
                placeholder="CC(=O)OC1=CC=CC=C1C(=O)O"
                value={smiles}
                onChange={(e) => setSmiles(e.target.value)}
                className="min-h-[100px] font-mono text-sm"
                required
              />
              <p className="text-xs text-muted-foreground">Enter a valid SMILES notation for the chemical structure</p>
            </div>

            <div className="space-y-2">
              <Label className="text-sm font-medium">Quick Examples</Label>
              <div className="flex flex-wrap gap-2">
                {exampleSmiles.map((example) => (
                  <Button
                    key={example.name}
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => setSmiles(example.smiles)}
                  >
                    {example.name}
                  </Button>
                ))}
              </div>
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <Button type="submit" className="w-full" disabled={loading || !smiles}>
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                "Predict Toxicity"
              )}
            </Button>
          </form>
        </CardContent>
      </Card>

      {result && <PredictionResult data={result} />}
    </div>
  )
}
