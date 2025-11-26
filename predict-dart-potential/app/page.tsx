import { PredictionForm } from "@/components/prediction-form"
import { BatchAnalysis } from "@/components/batch-analysis"
import { ModelInfo } from "@/components/model-info"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { FlaskConical, TrendingUp, Database } from "lucide-react"

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background via-background to-muted/20">
      {/* Header */}
      <header className="border-b bg-card/50 backdrop-blur supports-[backdrop-filter]:bg-card/50">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary text-primary-foreground">
              <FlaskConical className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight">DART Toxicity Predictor</h1>
              <p className="text-sm text-muted-foreground">AI-powered chemical safety analysis</p>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-12">
        <div className="mx-auto max-w-3xl text-center">
          <h2 className="text-balance text-4xl font-bold tracking-tight sm:text-5xl">
            Predict Chemical Toxicity with AI
          </h2>
          <p className="mt-4 text-pretty text-lg text-muted-foreground">
            Advanced machine learning models trained on ToxCast data to predict developmental and reproductive toxicity
            (DART) of chemical compounds
          </p>
        </div>

        {/* Feature Cards */}
        <div className="mx-auto mt-12 grid max-w-5xl gap-6 sm:grid-cols-3">
          <div className="rounded-xl border bg-card p-6 text-center shadow-sm">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <FlaskConical className="h-6 w-6" />
            </div>
            <h3 className="mt-4 font-semibold">SMILES Analysis</h3>
            <p className="mt-2 text-sm text-muted-foreground">
              Enter chemical structures using SMILES notation for instant predictions
            </p>
          </div>
          <div className="rounded-xl border bg-card p-6 text-center shadow-sm">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-lg bg-accent/10 text-accent">
              <TrendingUp className="h-6 w-6" />
            </div>
            <h3 className="mt-4 font-semibold">High Accuracy</h3>
            <p className="mt-2 text-sm text-muted-foreground">
              Models trained on comprehensive ToxCast dataset with validated performance
            </p>
          </div>
          <div className="rounded-xl border bg-card p-6 text-center shadow-sm">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-lg bg-chart-3/10 text-chart-3">
              <Database className="h-6 w-6" />
            </div>
            <h3 className="mt-4 font-semibold">Batch Processing</h3>
            <p className="mt-2 text-sm text-muted-foreground">
              Analyze multiple compounds simultaneously for efficient screening
            </p>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <section className="container mx-auto px-4 pb-12">
        <div className="mx-auto max-w-5xl">
          <Tabs defaultValue="single" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="single">Single Analysis</TabsTrigger>
              <TabsTrigger value="batch">Batch Analysis</TabsTrigger>
              <TabsTrigger value="model">Model Info</TabsTrigger>
            </TabsList>
            <TabsContent value="single" className="mt-6">
              <PredictionForm />
            </TabsContent>
            <TabsContent value="batch" className="mt-6">
              <BatchAnalysis />
            </TabsContent>
            <TabsContent value="model" className="mt-6">
              <ModelInfo />
            </TabsContent>
          </Tabs>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t bg-muted/30 py-8">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>Powered by ToxCast data and advanced machine learning</p>
          <p className="mt-2">For research purposes only. Not for clinical use.</p>
        </div>
      </footer>
    </div>
  )
}
