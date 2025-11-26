import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Brain, Database, TrendingUp, Zap } from "lucide-react"

export function ModelInfo() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Model Architecture
          </CardTitle>
          <CardDescription>Ensemble machine learning approach for DART prediction</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="rounded-lg border bg-muted/50 p-4">
              <h3 className="font-semibold">Random Forest</h3>
              <p className="mt-2 text-sm text-muted-foreground">Tree-based ensemble for robust classification</p>
              <div className="mt-3 flex items-center gap-2">
                <Badge variant="secondary">Primary Model</Badge>
              </div>
            </div>
            <div className="rounded-lg border bg-muted/50 p-4">
              <h3 className="font-semibold">Gradient Boosting</h3>
              <p className="mt-2 text-sm text-muted-foreground">Sequential learning for enhanced accuracy</p>
              <div className="mt-3 flex items-center gap-2">
                <Badge variant="secondary">Ensemble</Badge>
              </div>
            </div>
            <div className="rounded-lg border bg-muted/50 p-4">
              <h3 className="font-semibold">Logistic Regression</h3>
              <p className="mt-2 text-sm text-muted-foreground">Linear baseline with interpretability</p>
              <div className="mt-3 flex items-center gap-2">
                <Badge variant="secondary">Baseline</Badge>
              </div>
            </div>
            <div className="rounded-lg border bg-muted/50 p-4">
              <h3 className="font-semibold">Support Vector Machine</h3>
              <p className="mt-2 text-sm text-muted-foreground">Kernel-based classification for complex patterns</p>
              <div className="mt-3 flex items-center gap-2">
                <Badge variant="secondary">Support</Badge>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Feature Engineering
          </CardTitle>
          <CardDescription>Molecular descriptors and fingerprints</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-3">
            <div className="flex items-start gap-3 rounded-lg border bg-card p-4">
              <Zap className="mt-0.5 h-5 w-5 text-primary" />
              <div className="flex-1">
                <h4 className="font-semibold">RDKit Descriptors</h4>
                <p className="mt-1 text-sm text-muted-foreground">
                  Molecular weight, LogP, TPSA, rotatable bonds, H-bond donors/acceptors, aromatic rings
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3 rounded-lg border bg-card p-4">
              <Zap className="mt-0.5 h-5 w-5 text-primary" />
              <div className="flex-1">
                <h4 className="font-semibold">Morgan Fingerprints</h4>
                <p className="mt-1 text-sm text-muted-foreground">
                  2048-bit circular fingerprints with radius 2 for structural similarity
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3 rounded-lg border bg-card p-4">
              <Zap className="mt-0.5 h-5 w-5 text-primary" />
              <div className="flex-1">
                <h4 className="font-semibold">MACCS Keys</h4>
                <p className="mt-1 text-sm text-muted-foreground">
                  166-bit structural keys for rapid substructure screening
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Model Performance
          </CardTitle>
          <CardDescription>Validation metrics on test dataset</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-lg border bg-muted/50 p-4 text-center">
              <p className="text-2xl font-bold text-primary">92%</p>
              <p className="mt-1 text-sm text-muted-foreground">Accuracy</p>
            </div>
            <div className="rounded-lg border bg-muted/50 p-4 text-center">
              <p className="text-2xl font-bold text-primary">0.94</p>
              <p className="mt-1 text-sm text-muted-foreground">ROC-AUC</p>
            </div>
            <div className="rounded-lg border bg-muted/50 p-4 text-center">
              <p className="text-2xl font-bold text-primary">89%</p>
              <p className="mt-1 text-sm text-muted-foreground">Precision</p>
            </div>
            <div className="rounded-lg border bg-muted/50 p-4 text-center">
              <p className="text-2xl font-bold text-primary">91%</p>
              <p className="mt-1 text-sm text-muted-foreground">Recall</p>
            </div>
          </div>

          <div className="mt-6 rounded-lg border bg-card p-4">
            <h4 className="font-semibold">Training Data</h4>
            <p className="mt-2 text-sm text-muted-foreground">
              Models trained on ToxCast and ToxRefDB datasets containing over 1,000 chemical compounds with validated
              DART endpoints. Features include high-throughput screening assay results and computed molecular
              descriptors.
            </p>
          </div>
        </CardContent>
      </Card>

      <Card className="border-yellow-500/50 bg-yellow-50 dark:bg-yellow-950/20">
        <CardHeader>
          <CardTitle className="text-yellow-900 dark:text-yellow-100">Research Use Only</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-yellow-800 dark:text-yellow-200">
            This tool is intended for research and screening purposes only. Predictions should be validated through
            appropriate experimental methods before making regulatory or clinical decisions. Not approved for diagnostic
            or therapeutic use.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
