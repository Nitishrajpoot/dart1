"use client"

import { AlertDescription } from "@/components/ui/alert"

import { Alert } from "@/components/ui/alert"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { CheckCircle2, XCircle, AlertTriangle, TrendingUp } from "lucide-react"
import { cn } from "@/lib/utils"

interface PredictionResultProps {
  data: {
    prediction: string
    probability: number
    confidence: number
    model_used: string
    features_used: number
    smiles: string
  }
}

export function PredictionResult({ data }: PredictionResultProps) {
  const isToxic = data.prediction === "Toxic"
  const probabilityPercent = Math.round(data.probability * 100)
  const confidencePercent = Math.round(data.confidence * 100)

  const getRiskLevel = () => {
    if (probabilityPercent >= 75) return { label: "High Risk", color: "destructive" }
    if (probabilityPercent >= 50) return { label: "Moderate Risk", color: "warning" }
    return { label: "Low Risk", color: "success" }
  }

  const riskLevel = getRiskLevel()

  return (
    <Card className="border-2">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              {isToxic ? (
                <XCircle className="h-5 w-5 text-destructive" />
              ) : (
                <CheckCircle2 className="h-5 w-5 text-accent" />
              )}
              Prediction Result
            </CardTitle>
            <CardDescription className="mt-2">Analysis complete for compound</CardDescription>
          </div>
          <Badge
            variant={isToxic ? "destructive" : "default"}
            className={cn("text-sm", !isToxic && "bg-accent text-accent-foreground")}
          >
            {data.prediction}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* SMILES Display */}
        <div className="rounded-lg bg-muted p-4">
          <p className="text-xs font-medium text-muted-foreground">SMILES Notation</p>
          <p className="mt-1 break-all font-mono text-sm">{data.smiles}</p>
        </div>

        {/* Risk Assessment */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Risk Assessment</span>
            </div>
            <Badge variant="outline" className="text-xs">
              {riskLevel.label}
            </Badge>
          </div>

          {/* Toxicity Probability */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Toxicity Probability</span>
              <span className="font-semibold">{probabilityPercent}%</span>
            </div>
            <Progress
              value={probabilityPercent}
              className={cn(
                "h-2",
                probabilityPercent >= 75 && "[&>div]:bg-destructive",
                probabilityPercent >= 50 &&
                  probabilityPercent < 75 &&
                  "[&>div]:bg-yellow-500 dark:[&>div]:bg-yellow-600",
                probabilityPercent < 50 && "[&>div]:bg-accent",
              )}
            />
          </div>

          {/* Confidence */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Model Confidence</span>
              <span className="font-semibold">{confidencePercent}%</span>
            </div>
            <Progress value={confidencePercent} className="h-2 [&>div]:bg-primary" />
          </div>
        </div>

        {/* Model Information */}
        <div className="grid grid-cols-2 gap-4 rounded-lg border bg-card p-4">
          <div>
            <p className="text-xs text-muted-foreground">Model Used</p>
            <p className="mt-1 text-sm font-medium">{data.model_used}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Features Analyzed</p>
            <p className="mt-1 text-sm font-medium">{data.features_used}</p>
          </div>
        </div>

        {/* Warning Note */}
        {probabilityPercent >= 50 && (
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription className="text-sm">
              This compound shows potential DART toxicity. Further experimental validation is recommended.
            </AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  )
}
