# IMPULATOR-3 Production Architecture Plan

## Transforming from Prototype to Production-Ready Application

**Document Version**: 1.0
**Date**: November 2025
**Target Scale**: 100-1000 concurrent users
**Current State**: Single-user Streamlit prototype
**Goal**: Multi-user, scalable, production-grade application

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Architecture Analysis](#current-architecture-analysis)
3. [Proposed Production Architecture](#proposed-production-architecture)
4. [Backend Architecture](#backend-architecture)
5. [Frontend Architecture](#frontend-architecture)
6. [Queue & Job Processing System](#queue--job-processing-system)
7. [Database Design](#database-design)
8. [Infrastructure & Deployment](#infrastructure--deployment)
9. [Security & Authentication](#security--authentication)
10. [Monitoring & Observability](#monitoring--observability)
11. [Migration Strategy](#migration-strategy)
12. [Cost Estimation](#cost-estimation)
13. [Timeline & Roadmap](#timeline--roadmap)
14. [Risk Analysis & Mitigation](#risk-analysis--mitigation)

---

## Executive Summary

### Current Limitations

❌ **Single-user**: Streamlit session state not shared across users
❌ **No concurrency**: One compound processing at a time
❌ **No job queue**: Batch processing blocks UI
❌ **No persistence**: Data stored only in local filesystem
❌ **No scalability**: Cannot handle multiple simultaneous users
❌ **No monitoring**: No visibility into system health or job status
❌ **No authentication**: No user management or access control

### Proposed Solution

✅ **Multi-tenant**: Separate workspaces for each user/organization
✅ **Asynchronous processing**: Background job queue with workers
✅ **Horizontal scaling**: Auto-scaling workers based on queue depth
✅ **Cloud storage**: S3-compatible object storage for compounds
✅ **Real-time updates**: WebSocket connections for live progress
✅ **Full observability**: Metrics, logs, tracing, alerting
✅ **Enterprise auth**: SSO, RBAC, API keys

### Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                         USERS (100-1000)                        │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LOAD BALANCER (nginx)                       │
└───────────┬──────────────────────────────────┬──────────────────┘
            │                                  │
            ▼                                  ▼
┌───────────────────────────┐    ┌───────────────────────────────┐
│   FRONTEND (React + TS)   │    │   API GATEWAY (FastAPI)       │
│   - Modern SPA            │    │   - REST + GraphQL            │
│   - D3.js visualizations  │    │   - WebSocket support         │
│   - Three.js 3D molecules │    │   - Rate limiting             │
└───────────────────────────┘    └────────────┬──────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────┐
                    ▼                         ▼                     ▼
        ┌───────────────────┐   ┌───────────────────┐   ┌────────────────┐
        │  AUTHENTICATION   │   │   CORE API        │   │  WEBSOCKET     │
        │  (Keycloak/Auth0) │   │   (FastAPI)       │   │  SERVER        │
        └───────────────────┘   └─────────┬─────────┘   └────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
        ┌───────────────────┐  ┌───────────────────┐  ┌────────────────────┐
        │  MESSAGE QUEUE    │  │   DATABASE         │  │  OBJECT STORAGE    │
        │  (Redis/RabbitMQ) │  │   (PostgreSQL)     │  │  (MinIO/S3)        │
        └─────────┬─────────┘  └───────────────────┘  └────────────────────┘
                  │
                  ▼
        ┌─────────────────────────────────────────────┐
        │      WORKER POOL (Celery/Kubernetes Jobs)   │
        │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
        │  │ Worker 1 │  │ Worker 2 │  │ Worker N │  │
        │  └──────────┘  └──────────┘  └──────────┘  │
        │  - Auto-scaling based on queue depth        │
        │  - Independent processing of each compound  │
        │  - Fault-tolerant with retries              │
        └─────────────────────────────────────────────┘
```

---

## Current Architecture Analysis

### Technology Stack (Current)

| Component | Technology | Limitations |
|-----------|-----------|-------------|
| Frontend | Streamlit | Single-user, limited customization, slow for complex UIs |
| Backend | Python (synchronous) | Blocks during processing, no concurrency |
| Storage | Local filesystem | Not shared, no redundancy, ephemeral |
| Database | CSV files | No ACID, no concurrency control, data corruption risk |
| Queue | None | No job management |
| Deployment | Single server | No scaling, single point of failure |

### Bottlenecks Identified

1. **Processing Pipeline**
   - ChEMBL API queries: 2-5 seconds per compound
   - PDB queries: 1-2 seconds per compound
   - RDKit calculations: 0.5-1 second per compound
   - Total per compound: **4-8 seconds minimum**
   - Batch of 100 compounds: **6-13 minutes** (blocking)

2. **Storage**
   - CSV files lock during write
   - No concurrent access
   - No backup/versioning

3. **Compute**
   - Single-threaded processing
   - CPU-bound operations (RDKit, numpy)
   - Memory leaks in long sessions

---

## Proposed Production Architecture

### Design Principles

1. **Separation of Concerns**: Frontend, API, Workers, Storage
2. **Asynchronous by Default**: Non-blocking operations
3. **Horizontal Scalability**: Add more workers/API servers as needed
4. **Fault Tolerance**: Retries, circuit breakers, graceful degradation
5. **Observability**: Logs, metrics, traces for every component
6. **Security First**: Authentication, authorization, encryption, audit logs

### Technology Stack (Proposed)

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **Frontend** | React 18 + TypeScript | Modern, fast, large ecosystem |
| **Visualization** | D3.js + Three.js | Best-in-class for 2D/3D molecular viz |
| **State Management** | Redux Toolkit + RTK Query | Predictable state, built-in caching |
| **UI Components** | Material-UI or Chakra UI | Professional, accessible, customizable |
| **API Gateway** | FastAPI (Python) | High performance, auto docs, async native |
| **Message Queue** | Redis + Celery | Battle-tested, Python-native, easy scaling |
| **Workers** | Celery Workers (Python) | Reuse existing codebase, proven at scale |
| **Database** | PostgreSQL 15+ | ACID, JSON support, full-text search |
| **Object Storage** | MinIO (S3-compatible) | Self-hosted, cost-effective, S3 API |
| **Cache** | Redis | Fast lookups, session storage |
| **Auth** | Keycloak or Auth0 | SSO, OIDC, SAML, RBAC |
| **Orchestration** | Kubernetes (k8s) | Auto-scaling, self-healing, industry standard |
| **Monitoring** | Prometheus + Grafana | Metrics, dashboards, alerting |
| **Logging** | ELK Stack (Elasticsearch, Logstash, Kibana) | Centralized logs, search, analysis |
| **Tracing** | Jaeger or OpenTelemetry | Distributed tracing, performance analysis |

---

## Backend Architecture

### API Layer (FastAPI)

#### Core Responsibilities

1. **Authentication & Authorization**
   - JWT token validation
   - User/organization isolation
   - Role-based access control (RBAC)

2. **Request Handling**
   - REST endpoints for CRUD operations
   - GraphQL for complex queries
   - WebSocket for real-time updates

3. **Job Submission**
   - Validate input (SMILES, InChI, CSV)
   - Create job records in database
   - Submit to message queue
   - Return job ID to client

4. **Data Retrieval**
   - Query compound metadata
   - Fetch processing results
   - Stream large datasets
   - Generate exports (CSV, JSON, PDF)

#### API Endpoints

**Compounds API**

```python
# POST /api/v1/compounds - Submit single compound
{
  "name": "Compound A",
  "structure_input": "SMILES_STRING",
  "input_format": "smiles",
  "similarity_threshold": 0.85,
  "activity_types": ["IC50", "Ki"]
}

Response: {
  "job_id": "uuid-1234",
  "status": "queued",
  "position": 5,
  "estimated_time": 45  # seconds
}

# POST /api/v1/compounds/batch - Submit CSV batch
{
  "file": "base64_encoded_csv",
  "options": {...}
}

Response: {
  "batch_id": "batch-uuid-5678",
  "job_ids": ["job-1", "job-2", ..., "job-10"],
  "total_compounds": 10,
  "status": "queued"
}

# GET /api/v1/compounds/{compound_id} - Get compound details
Response: {
  "id": "cmp-123",
  "name": "Compound A",
  "smiles": "...",
  "status": "completed",
  "oqpla_score": 0.75,
  "created_at": "2025-11-19T...",
  "results": {...}
}

# GET /api/v1/compounds - List all compounds (paginated)
?page=1&limit=50&sort=oqpla_score&order=desc&filter=status:completed

# DELETE /api/v1/compounds/{compound_id} - Delete compound
```

**Jobs API**

```python
# GET /api/v1/jobs/{job_id} - Get job status
Response: {
  "job_id": "uuid-1234",
  "status": "processing",  # queued, processing, completed, failed
  "progress": 65,  # percentage
  "current_step": "Calculating O[Q/P/L]A scores",
  "created_at": "...",
  "started_at": "...",
  "completed_at": null,
  "result": null,
  "error": null
}

# GET /api/v1/jobs - List user's jobs
?status=processing&limit=20

# POST /api/v1/jobs/{job_id}/cancel - Cancel job
# POST /api/v1/jobs/{job_id}/retry - Retry failed job
```

**Analytics API**

```python
# GET /api/v1/analytics/summary - Dashboard summary
Response: {
  "total_compounds": 1523,
  "strong_imps": 45,
  "processing_today": 12,
  "avg_oqpla_score": 0.62
}

# GET /api/v1/analytics/distribution - Score distribution
# GET /api/v1/analytics/trends - Time-series trends
```

**Export API**

```python
# POST /api/v1/export/csv - Export compounds as CSV
{
  "compound_ids": ["cmp-1", "cmp-2", ...],
  "columns": ["name", "smiles", "oqpla_score", ...]
}

# POST /api/v1/export/pdf - Generate PDF report
# POST /api/v1/export/sdf - Export as SDF file
```

#### GraphQL Schema

```graphql
type Compound {
  id: ID!
  name: String!
  smiles: String!
  chemblId: String
  oqplaScore: Float
  oqplaClassification: String
  status: CompoundStatus!
  createdAt: DateTime!
  bioactivities: [Bioactivity!]!
  interferences: InterferenceFlags!
  pdbEvidence: PDBEvidence
}

type Bioactivity {
  id: ID!
  targetName: String!
  activityType: String!
  pActivity: Float!
  sei: Float
  bei: Float
  nsei: Float
  nbei: Float
}

type Query {
  compound(id: ID!): Compound
  compounds(
    filter: CompoundFilter
    sort: CompoundSort
    pagination: Pagination
  ): CompoundConnection!
  searchCompounds(query: String!): [Compound!]!
  job(id: ID!): Job!
  jobs(status: JobStatus): [Job!]!
}

type Mutation {
  submitCompound(input: CompoundInput!): Job!
  submitBatch(input: BatchInput!): BatchJob!
  deleteCompound(id: ID!): Boolean!
  cancelJob(id: ID!): Boolean!
}

type Subscription {
  jobUpdated(jobId: ID!): Job!
  compoundProcessed(userId: ID!): Compound!
}
```

---

## Frontend Architecture

### Technology Stack

**Core Framework**: React 18 + TypeScript

**State Management**:
- Redux Toolkit for global state
- RTK Query for API caching
- React Query (alternative) for server state

**Routing**: React Router v6

**UI Components**: Material-UI (MUI) or Chakra UI

**Data Visualization**:
- **D3.js**: 2D charts, plots, efficiency planes
- **Plotly.js**: Interactive scatter plots, boxplots
- **Three.js**: 3D molecular structures
- **Recharts**: Simple charts and dashboards

**Molecular Rendering**:
- **3Dmol.js**: WebGL-based molecular viewer (current)
- **NGL Viewer**: High-performance alternative
- **RDKit.js**: Client-side SMILES rendering

**Form Handling**: React Hook Form + Yup validation

**File Upload**: react-dropzone

**Notifications**: react-toastify or notistack

**WebSocket**: Socket.io-client for real-time updates

### Application Structure

```
frontend/
├── public/
│   ├── index.html
│   └── assets/
├── src/
│   ├── api/                    # API client & types
│   │   ├── client.ts           # Axios/fetch wrapper
│   │   ├── compounds.api.ts
│   │   ├── jobs.api.ts
│   │   └── types.ts
│   ├── components/             # Reusable components
│   │   ├── common/
│   │   │   ├── Button.tsx
│   │   │   ├── Card.tsx
│   │   │   └── ...
│   │   ├── compounds/
│   │   │   ├── CompoundCard.tsx
│   │   │   ├── CompoundTable.tsx
│   │   │   └── CompoundForm.tsx
│   │   ├── visualizations/
│   │   │   ├── EfficiencyPlane.tsx     # D3.js
│   │   │   ├── MoleculeViewer.tsx      # Three.js
│   │   │   ├── OQPLAChart.tsx          # Recharts
│   │   │   └── ScoreDistribution.tsx
│   │   └── jobs/
│   │       ├── JobProgress.tsx
│   │       └── JobQueue.tsx
│   ├── features/               # Feature-based modules
│   │   ├── auth/
│   │   │   ├── Login.tsx
│   │   │   ├── authSlice.ts
│   │   │   └── authAPI.ts
│   │   ├── compounds/
│   │   │   ├── CompoundList.tsx
│   │   │   ├── CompoundDetails.tsx
│   │   │   ├── CompoundSubmit.tsx
│   │   │   ├── compoundsSlice.ts
│   │   │   └── compoundsAPI.ts
│   │   ├── analytics/
│   │   │   ├── Dashboard.tsx
│   │   │   └── analyticsSlice.ts
│   │   └── jobs/
│   │       ├── JobMonitor.tsx
│   │       └── jobsSlice.ts
│   ├── hooks/                  # Custom React hooks
│   │   ├── useCompounds.ts
│   │   ├── useWebSocket.ts
│   │   ├── useJobStatus.ts
│   │   └── useAuth.ts
│   ├── layouts/
│   │   ├── MainLayout.tsx
│   │   ├── DashboardLayout.tsx
│   │   └── AuthLayout.tsx
│   ├── pages/
│   │   ├── HomePage.tsx
│   │   ├── CompoundsPage.tsx
│   │   ├── CompoundDetailsPage.tsx
│   │   ├── AnalyticsPage.tsx
│   │   ├── SubmitPage.tsx
│   │   └── NotFoundPage.tsx
│   ├── store/
│   │   ├── index.ts            # Redux store config
│   │   └── rootReducer.ts
│   ├── utils/
│   │   ├── formatters.ts
│   │   ├── validators.ts
│   │   └── constants.ts
│   ├── App.tsx
│   ├── index.tsx
│   └── theme.ts
├── package.json
├── tsconfig.json
└── vite.config.ts              # Using Vite for fast builds
```

### Key Features

#### 1. Real-Time Job Progress

```typescript
// useJobStatus.ts
import { useEffect, useState } from 'react';
import io from 'socket.io-client';

export function useJobStatus(jobId: string) {
  const [status, setStatus] = useState<JobStatus>('queued');
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const socket = io('ws://api.impulator.com');

    socket.emit('subscribe_job', { jobId });

    socket.on('job_updated', (data) => {
      setStatus(data.status);
      setProgress(data.progress);
    });

    return () => {
      socket.disconnect();
    };
  }, [jobId]);

  return { status, progress };
}

// JobProgress.tsx
function JobProgress({ jobId }: { jobId: string }) {
  const { status, progress } = useJobStatus(jobId);

  return (
    <Box>
      <Typography>{getStatusLabel(status)}</Typography>
      <LinearProgress variant="determinate" value={progress} />
      <Typography>{progress}%</Typography>
    </Box>
  );
}
```

#### 2. Interactive 3D Molecule Viewer

```typescript
// MoleculeViewer.tsx
import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

interface MoleculeViewerProps {
  pdbData: string;
  style?: 'stick' | 'sphere' | 'cartoon';
}

export function MoleculeViewer({ pdbData, style = 'stick' }: MoleculeViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Three.js scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });

    // Parse PDB and create molecule
    const molecule = parsePDB(pdbData);
    scene.add(createMoleculeGeometry(molecule, style));

    // Add controls
    const controls = new OrbitControls(camera, renderer.domElement);

    // Animation loop
    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    return () => {
      renderer.dispose();
    };
  }, [pdbData, style]);

  return <div ref={containerRef} style={{ width: '100%', height: '500px' }} />;
}
```

#### 3. Efficiency Plane Visualization (D3.js)

```typescript
// EfficiencyPlane.tsx
import { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface DataPoint {
  sei: number;
  bei: number;
  name: string;
  oqplaScore: number;
}

export function EfficiencyPlane({ data }: { data: DataPoint[] }) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const width = 800;
    const height = 600;
    const margin = { top: 20, right: 20, bottom: 50, left: 60 };

    // Scales
    const xScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.sei)!])
      .range([margin.left, width - margin.right]);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.bei)!])
      .range([height - margin.bottom, margin.top]);

    const colorScale = d3.scaleSequential()
      .domain([0, 1])
      .interpolator(d3.interpolateViridis);

    // Clear previous
    svg.selectAll('*').remove();

    // Axes
    svg.append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(xScale));

    svg.append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(yScale));

    // Points
    svg.selectAll('circle')
      .data(data)
      .join('circle')
      .attr('cx', d => xScale(d.sei))
      .attr('cy', d => yScale(d.bei))
      .attr('r', 5)
      .attr('fill', d => colorScale(d.oqplaScore))
      .attr('opacity', 0.7)
      .on('mouseover', function(event, d) {
        // Tooltip
        d3.select(this).attr('r', 8);
      })
      .on('mouseout', function() {
        d3.select(this).attr('r', 5);
      });

    // Labels
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 10)
      .attr('text-anchor', 'middle')
      .text('SEI (Surface Efficiency Index)');

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .text('BEI (Binding Efficiency Index)');

  }, [data]);

  return <svg ref={svgRef} width={800} height={600} />;
}
```

#### 4. Compound Submission Form

```typescript
// CompoundSubmit.tsx
import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';

const schema = yup.object({
  name: yup.string().required('Compound name is required'),
  structureInput: yup.string().required('Structure is required'),
  inputFormat: yup.string().oneOf(['smiles', 'inchi']).required(),
  similarityThreshold: yup.number().min(0.5).max(1.0).default(0.85),
  activityTypes: yup.array().of(yup.string()).min(1)
});

type FormData = yup.InferType<typeof schema>;

export function CompoundSubmit() {
  const { register, handleSubmit, formState: { errors } } = useForm<FormData>({
    resolver: yupResolver(schema)
  });

  const [submitCompound, { isLoading }] = useSubmitCompoundMutation();

  const onSubmit = async (data: FormData) => {
    try {
      const result = await submitCompound(data).unwrap();
      toast.success(`Job submitted: ${result.job_id}`);
      navigate(`/jobs/${result.job_id}`);
    } catch (error) {
      toast.error('Submission failed');
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <TextField
        {...register('name')}
        label="Compound Name"
        error={!!errors.name}
        helperText={errors.name?.message}
      />

      <TextField
        {...register('structureInput')}
        label="SMILES/InChI"
        multiline
        rows={3}
        error={!!errors.structureInput}
        helperText={errors.structureInput?.message}
      />

      <FormControl>
        <InputLabel>Activity Types</InputLabel>
        <Select {...register('activityTypes')} multiple>
          <MenuItem value="IC50">IC50</MenuItem>
          <MenuItem value="Ki">Ki</MenuItem>
          <MenuItem value="Kd">Kd</MenuItem>
        </Select>
      </FormControl>

      <Button type="submit" disabled={isLoading}>
        {isLoading ? <CircularProgress size={24} /> : 'Submit'}
      </Button>
    </form>
  );
}
```

---

## Queue & Job Processing System

### Architecture

```
┌──────────────────────┐
│   API Server         │
│   (FastAPI)          │
└─────────┬────────────┘
          │ Submit Job
          ▼
┌──────────────────────────────────────────┐
│         Redis Queue                      │
│  ┌────────┐  ┌────────┐  ┌────────┐     │
│  │ Job 1  │→ │ Job 2  │→ │ Job 3  │ ... │
│  └────────┘  └────────┘  └────────┘     │
│  Priority: High → Medium → Low          │
└──────────┬───────────────────────────────┘
           │ Fetch Job (FIFO)
           ▼
┌────────────────────────────────────────────────┐
│         Worker Pool (Celery)                   │
│  ┌──────────────┐  ┌──────────────┐           │
│  │   Worker 1   │  │   Worker 2   │  ...      │
│  │  Processing  │  │     Idle     │           │
│  └──────────────┘  └──────────────┘           │
│  Auto-scale: 1-20 workers based on queue      │
└────────────┬───────────────────────────────────┘
             │ Update Status
             ▼
┌──────────────────────────────────────────┐
│         PostgreSQL Database              │
│  jobs table: status, progress, result    │
└──────────────────────────────────────────┘
```

### Implementation

#### Job Model (Database)

```sql
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    job_type VARCHAR(50) NOT NULL,  -- 'single_compound', 'batch'
    status VARCHAR(20) NOT NULL,    -- 'queued', 'processing', 'completed', 'failed'
    priority INTEGER DEFAULT 5,     -- 1 (high) to 10 (low)
    progress INTEGER DEFAULT 0,     -- 0-100
    current_step TEXT,
    input_data JSONB NOT NULL,
    result_data JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    retries INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    parent_job_id UUID REFERENCES jobs(id),  -- For batch jobs
    INDEX idx_user_status (user_id, status),
    INDEX idx_created_at (created_at),
    INDEX idx_parent_job (parent_job_id)
);

CREATE TABLE compounds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    job_id UUID REFERENCES jobs(id),
    name VARCHAR(255) NOT NULL,
    smiles TEXT NOT NULL,
    chembl_id VARCHAR(50),
    status VARCHAR(20) NOT NULL,  -- 'processing', 'completed', 'failed'
    oqpla_score FLOAT,
    oqpla_classification VARCHAR(50),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_user_status (user_id, status),
    INDEX idx_oqpla_score (oqpla_score DESC),
    INDEX idx_chembl_id (chembl_id)
);
```

#### Celery Configuration

```python
# celery_config.py
from celery import Celery
from kombu import Queue, Exchange

app = Celery('impulator', broker='redis://redis:6379/0')

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Task routing
    task_routes={
        'tasks.process_compound': {'queue': 'compounds'},
        'tasks.process_batch': {'queue': 'batches'},
        'tasks.export_results': {'queue': 'exports'},
    },

    # Queue configuration
    task_queues=(
        Queue('compounds', Exchange('compounds'), routing_key='compound',
              priority_steps=[1, 3, 5, 7, 9]),  # Priority levels
        Queue('batches', Exchange('batches'), routing_key='batch'),
        Queue('exports', Exchange('exports'), routing_key='export'),
    ),

    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time
    worker_max_tasks_per_child=100,  # Restart after 100 tasks (prevent memory leaks)
    task_acks_late=True,  # Acknowledge after task completes
    task_reject_on_worker_lost=True,  # Re-queue if worker dies

    # Retry settings
    task_default_retry_delay=60,  # Wait 60s before retry
    task_max_retries=3,

    # Result backend
    result_backend='redis://redis:6379/1',
    result_expires=3600,  # Results expire after 1 hour
)

# Auto-scaling configuration
app.conf.worker_autoscaler = 'celery.worker.autoscale:Autoscaler'
app.conf.worker_max_concurrency = 20
app.conf.worker_min_concurrency = 1
```

#### Task Definition

```python
# tasks.py
from celery import Task, group, chord
from celery.utils.log import get_task_logger
from typing import Dict, Any

logger = get_task_logger(__name__)

class CallbackTask(Task):
    """Base task with callbacks for status updates."""

    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds."""
        update_job_status(task_id, 'completed', result=retval)
        notify_user_via_websocket(task_id, 'completed')

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails."""
        update_job_status(task_id, 'failed', error=str(exc))
        notify_user_via_websocket(task_id, 'failed')

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried."""
        update_job_status(task_id, 'retrying', error=str(exc))

@app.task(base=CallbackTask, bind=True, max_retries=3)
def process_compound(self, job_id: str, compound_data: Dict[str, Any]):
    """
    Process a single compound.

    Steps:
    1. Validate input
    2. Query ChEMBL
    3. Calculate efficiency metrics
    4. Query PDB (if enabled)
    5. Calculate O[Q/P/L]A score
    6. Store results
    """
    try:
        # Update status
        update_job_status(job_id, 'processing', progress=0)

        # Step 1: Validate
        logger.info(f"Validating compound {compound_data['name']}")
        validated = validate_compound_input(compound_data)
        update_job_status(job_id, 'processing', progress=10,
                         current_step='Validation complete')

        # Step 2: ChEMBL query
        logger.info(f"Querying ChEMBL for {validated['smiles']}")
        bioactivities = query_chembl(validated['smiles'],
                                     validated['similarity_threshold'])
        update_job_status(job_id, 'processing', progress=30,
                         current_step=f'Found {len(bioactivities)} bioactivities')

        if not bioactivities:
            raise ValueError("No bioactivities found in ChEMBL")

        # Step 3: Calculate efficiency metrics
        logger.info("Calculating efficiency metrics")
        df_bioactivities = calculate_efficiency_metrics(bioactivities)
        update_job_status(job_id, 'processing', progress=50,
                         current_step='Efficiency metrics calculated')

        # Step 4: PDB evidence (optional)
        pdb_score = 0.0
        if config.USE_PDB_EVIDENCE:
            logger.info("Querying PDB")
            pdb_score = query_pdb_evidence(validated['smiles'])
            update_job_status(job_id, 'processing', progress=70,
                             current_step='PDB evidence retrieved')

        # Step 5: O[Q/P/L]A scoring
        logger.info("Calculating O[Q/P/L]A scores")
        df_scored = calculate_oqpla_phase2(df_bioactivities, pdb_score)
        update_job_status(job_id, 'processing', progress=85,
                         current_step='O[Q/P/L]A scores calculated')

        # Step 6: Store results
        logger.info("Storing results")
        compound_id = store_compound_results(
            job_id=job_id,
            compound_data=validated,
            bioactivities=df_scored,
            oqpla_score=df_scored['OQPLA_Final_Score'].max()
        )

        update_job_status(job_id, 'processing', progress=100,
                         current_step='Results stored')

        return {
            'compound_id': compound_id,
            'oqpla_score': float(df_scored['OQPLA_Final_Score'].max()),
            'num_bioactivities': len(df_scored)
        }

    except Exception as e:
        logger.error(f"Error processing compound: {str(e)}", exc_info=True)

        # Retry logic
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60 * (self.request.retries + 1))

        raise

@app.task(base=CallbackTask, bind=True)
def process_batch(self, batch_job_id: str, compounds: list):
    """
    Process a batch of compounds.

    Creates individual jobs for each compound and tracks completion.
    """
    # Create child jobs
    child_jobs = []
    for compound in compounds:
        child_job = create_job(
            user_id=self.request.user_id,
            job_type='single_compound',
            input_data=compound,
            parent_job_id=batch_job_id
        )
        child_jobs.append(child_job.id)

    # Submit all child jobs in parallel
    job_group = group(
        process_compound.s(job_id, compound)
        for job_id, compound in zip(child_jobs, compounds)
    )

    # Use chord to execute callback when all complete
    callback = batch_complete_callback.s(batch_job_id)
    chord(job_group)(callback)

    return {'child_jobs': child_jobs, 'total': len(child_jobs)}

@app.task
def batch_complete_callback(results, batch_job_id):
    """Called when all batch jobs complete."""
    successful = sum(1 for r in results if r.get('compound_id'))
    failed = len(results) - successful

    update_job_status(
        batch_job_id,
        'completed',
        result={
            'total': len(results),
            'successful': successful,
            'failed': failed
        }
    )

    notify_user_via_websocket(batch_job_id, 'batch_completed')
```

#### Job Submission (API)

```python
# api/routes/compounds.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from tasks import process_compound, process_batch

router = APIRouter(prefix='/api/v1/compounds')

@router.post('/')
async def submit_compound(
    compound: CompoundInput,
    user: User = Depends(get_current_user)
):
    """Submit a single compound for processing."""

    # Create job record
    job = create_job(
        user_id=user.id,
        job_type='single_compound',
        input_data=compound.dict(),
        priority=compound.priority or 5
    )

    # Submit to Celery
    process_compound.apply_async(
        args=[str(job.id), compound.dict()],
        task_id=str(job.id),
        priority=compound.priority or 5
    )

    # Get queue position
    queue_position = get_queue_position(job.id)

    return {
        'job_id': str(job.id),
        'status': 'queued',
        'position': queue_position,
        'estimated_time': queue_position * 6  # 6 seconds per compound
    }

@router.post('/batch')
async def submit_batch(
    file: UploadFile,
    user: User = Depends(get_current_user)
):
    """Submit a CSV batch for processing."""

    # Parse CSV
    df = pd.read_csv(file.file)
    compounds = df.to_dict('records')

    # Validate
    if len(compounds) > 1000:
        raise HTTPException(400, "Max 1000 compounds per batch")

    # Create parent job
    batch_job = create_job(
        user_id=user.id,
        job_type='batch',
        input_data={'num_compounds': len(compounds)}
    )

    # Submit batch task
    process_batch.apply_async(
        args=[str(batch_job.id), compounds],
        task_id=str(batch_job.id)
    )

    return {
        'batch_id': str(batch_job.id),
        'total_compounds': len(compounds),
        'status': 'queued'
    }
```

#### Auto-Scaling Logic

```python
# autoscaler.py
import subprocess
import psutil
from celery import Celery

app = Celery('impulator')

def get_queue_depth():
    """Get number of pending jobs."""
    with app.connection() as conn:
        queue = conn.default_channel.queue_declare(
            queue='compounds', passive=True
        )
        return queue.message_count

def get_active_workers():
    """Get number of active Celery workers."""
    stats = app.control.inspect().stats()
    return len(stats) if stats else 0

def scale_workers():
    """Auto-scale workers based on queue depth."""
    queue_depth = get_queue_depth()
    active_workers = get_active_workers()

    # Scaling policy
    if queue_depth > 100 and active_workers < 20:
        # Scale up: 1 worker per 10 queued jobs
        target_workers = min(20, queue_depth // 10)
        scale_to(target_workers)

    elif queue_depth < 10 and active_workers > 2:
        # Scale down
        target_workers = max(2, queue_depth // 5)
        scale_to(target_workers)

def scale_to(target: int):
    """Scale Kubernetes deployment to target replicas."""
    subprocess.run([
        'kubectl', 'scale',
        '--replicas', str(target),
        'deployment/impulator-worker'
    ])

# Run every 30 seconds
if __name__ == '__main__':
    import schedule
    schedule.every(30).seconds.do(scale_workers)

    while True:
        schedule.run_pending()
        time.sleep(1)
```

---

## Database Design

### Schema

```sql
-- Users & Organizations
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    plan VARCHAR(50) DEFAULT 'free',  -- 'free', 'pro', 'enterprise'
    max_compounds INTEGER DEFAULT 100,
    max_concurrent_jobs INTEGER DEFAULT 5,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'user',  -- 'user', 'admin', 'viewer'
    api_key VARCHAR(64) UNIQUE,
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    INDEX idx_org (organization_id),
    INDEX idx_email (email)
);

-- Jobs (already defined above)

-- Compounds (already defined above)

-- Bioactivities
CREATE TABLE bioactivities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    compound_id UUID NOT NULL REFERENCES compounds(id) ON DELETE CASCADE,
    chembl_id VARCHAR(50),
    target_name VARCHAR(255),
    target_chembl_id VARCHAR(50),
    activity_type VARCHAR(50),
    pactivity FLOAT,
    sei FLOAT,
    bei FLOAT,
    nsei FLOAT,
    nbei FLOAT,
    modulus_sei_bei FLOAT,
    angle_sei_bei FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_compound (compound_id),
    INDEX idx_target (target_chembl_id),
    INDEX idx_pactivity (pactivity DESC)
);

-- Interference Flags
CREATE TABLE interference_flags (
    compound_id UUID PRIMARY KEY REFERENCES compounds(id) ON DELETE CASCADE,
    pains BOOLEAN DEFAULT FALSE,
    aggregator BOOLEAN DEFAULT FALSE,
    redox BOOLEAN DEFAULT FALSE,
    fluorescence BOOLEAN DEFAULT FALSE,
    thiol_reactive BOOLEAN DEFAULT FALSE,
    assay_quality_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- PDB Evidence
CREATE TABLE pdb_evidence (
    compound_id UUID PRIMARY KEY REFERENCES compounds(id) ON DELETE CASCADE,
    pdb_score FLOAT,
    num_structures INTEGER,
    high_quality_count INTEGER,
    medium_quality_count INTEGER,
    poor_quality_count INTEGER,
    pdb_ids TEXT[],  -- Array of PDB IDs
    best_resolution FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Exports
CREATE TABLE exports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    export_type VARCHAR(50),  -- 'csv', 'pdf', 'sdf'
    status VARCHAR(20),
    file_path TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    INDEX idx_user_created (user_id, created_at)
);

-- Audit Log
CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    action VARCHAR(100),
    resource_type VARCHAR(50),
    resource_id UUID,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_user_created (user_id, created_at),
    INDEX idx_resource (resource_type, resource_id)
);

-- System Metrics
CREATE TABLE system_metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_name VARCHAR(100),
    metric_value FLOAT,
    tags JSONB,
    recorded_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_name_time (metric_name, recorded_at)
);
```

### Query Optimization

```sql
-- Materialized view for dashboard statistics
CREATE MATERIALIZED VIEW user_statistics AS
SELECT
    u.id AS user_id,
    COUNT(DISTINCT c.id) AS total_compounds,
    COUNT(DISTINCT CASE WHEN c.status = 'completed' THEN c.id END) AS completed_compounds,
    AVG(c.oqpla_score) AS avg_oqpla_score,
    COUNT(DISTINCT CASE WHEN c.oqpla_classification = 'Strong IMP' THEN c.id END) AS strong_imps,
    MAX(c.created_at) AS last_activity
FROM users u
LEFT JOIN compounds c ON c.user_id = u.id
GROUP BY u.id;

-- Refresh every hour
CREATE INDEX idx_user_stats ON user_statistics(user_id);

-- Compound search index (full-text)
CREATE INDEX idx_compound_search ON compounds
USING gin(to_tsvector('english', name || ' ' || COALESCE(chembl_id, '')));

-- Query: Search compounds
SELECT * FROM compounds
WHERE to_tsvector('english', name || ' ' || COALESCE(chembl_id, ''))
      @@ plainto_tsquery('english', 'quercetin');
```

### Data Retention Policy

```sql
-- Delete old jobs after 30 days
DELETE FROM jobs
WHERE status IN ('completed', 'failed')
  AND completed_at < NOW() - INTERVAL '30 days';

-- Archive old audit logs to S3
-- (Run monthly via cron job)
```

---

## Infrastructure & Deployment

### Kubernetes Architecture

```yaml
# kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: impulator-prod
---
# kubernetes/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: impulator-config
  namespace: impulator-prod
data:
  DATABASE_URL: "postgresql://user:pass@postgres:5432/impulator"
  REDIS_URL: "redis://redis:6379/0"
  S3_ENDPOINT: "http://minio:9000"
  S3_BUCKET: "impulator-compounds"
  CHEMBL_API_URL: "https://www.ebi.ac.uk/chembl/api/data"
  PDB_API_URL: "https://data.rcsb.org/rest/v1"
  USE_PDB_EVIDENCE: "true"
---
# kubernetes/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: impulator-secrets
  namespace: impulator-prod
type: Opaque
data:
  DATABASE_PASSWORD: <base64>
  JWT_SECRET: <base64>
  S3_ACCESS_KEY: <base64>
  S3_SECRET_KEY: <base64>
---
# kubernetes/frontend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: impulator-frontend
  namespace: impulator-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: impulator-frontend
  template:
    metadata:
      labels:
        app: impulator-frontend
    spec:
      containers:
      - name: frontend
        image: impulator/frontend:1.0.0
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 80
          initialDelaySeconds: 10
          periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: impulator-frontend
  namespace: impulator-prod
spec:
  selector:
    app: impulator-frontend
  ports:
  - port: 80
    targetPort: 80
  type: LoadBalancer
---
# kubernetes/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: impulator-api
  namespace: impulator-prod
spec:
  replicas: 5
  selector:
    matchLabels:
      app: impulator-api
  template:
    metadata:
      labels:
        app: impulator-api
    spec:
      containers:
      - name: api
        image: impulator/api:1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: impulator-config
              key: DATABASE_URL
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: impulator-secrets
              key: DATABASE_PASSWORD
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: impulator-api
  namespace: impulator-prod
spec:
  selector:
    app: impulator-api
  ports:
  - port: 8000
    targetPort: 8000
---
# kubernetes/worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: impulator-worker
  namespace: impulator-prod
spec:
  replicas: 2  # Auto-scaled by HPA
  selector:
    matchLabels:
      app: impulator-worker
  template:
    metadata:
      labels:
        app: impulator-worker
    spec:
      containers:
      - name: worker
        image: impulator/worker:1.0.0
        command: ["celery", "-A", "tasks", "worker", "--loglevel=info"]
        env:
        - name: CELERY_BROKER_URL
          valueFrom:
            configMapKeyRef:
              name: impulator-config
              key: REDIS_URL
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
# kubernetes/worker-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: impulator-worker-hpa
  namespace: impulator-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: impulator-worker
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: External
    external:
      metric:
        name: redis_queue_depth
        selector:
          matchLabels:
            queue: compounds
      target:
        type: AverageValue
        averageValue: "10"  # 1 worker per 10 queued jobs
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
---
# kubernetes/redis-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: impulator-prod
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-data
          mountPath: /data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: redis-data
        persistentVolumeClaim:
          claimName: redis-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: impulator-prod
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
---
# kubernetes/postgres-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: impulator-prod
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: impulator
        - name: POSTGRES_USER
          value: impulator
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: impulator-secrets
              key: DATABASE_PASSWORD
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: impulator-prod
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  clusterIP: None  # Headless service for StatefulSet
---
# kubernetes/minio-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minio
  namespace: impulator-prod
spec:
  replicas: 1
  selector:
    matchLabels:
      app: minio
  template:
    metadata:
      labels:
        app: minio
    spec:
      containers:
      - name: minio
        image: minio/minio:latest
        args:
        - server
        - /data
        - --console-address
        - ":9001"
        ports:
        - containerPort: 9000
        - containerPort: 9001
        env:
        - name: MINIO_ROOT_USER
          valueFrom:
            secretKeyRef:
              name: impulator-secrets
              key: S3_ACCESS_KEY
        - name: MINIO_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: impulator-secrets
              key: S3_SECRET_KEY
        volumeMounts:
        - name: minio-data
          mountPath: /data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: minio-data
        persistentVolumeClaim:
          claimName: minio-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: minio
  namespace: impulator-prod
spec:
  selector:
    app: minio
  ports:
  - name: api
    port: 9000
    targetPort: 9000
  - name: console
    port: 9001
    targetPort: 9001
```

### Docker Images

**Frontend Dockerfile**

```dockerfile
# frontend/Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**API Dockerfile**

```dockerfile
# api/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run migrations and start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Worker Dockerfile**

```dockerfile
# worker/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (RDKit requires)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libcairo2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run Celery worker
CMD ["celery", "-A", "tasks", "worker", "--loglevel=info", "--concurrency=4"]
```

---

## Security & Authentication

### Authentication Flow

```
1. User logs in via Frontend
   ↓
2. Frontend sends credentials to Auth Service (Keycloak/Auth0)
   ↓
3. Auth Service validates and returns JWT token
   ↓
4. Frontend stores JWT in localStorage/sessionStorage
   ↓
5. Frontend includes JWT in Authorization header for API requests
   ↓
6. API validates JWT signature and expiration
   ↓
7. API extracts user_id from JWT claims
   ↓
8. API checks permissions (RBAC)
   ↓
9. API processes request with user context
```

### JWT Structure

```json
{
  "header": {
    "alg": "RS256",
    "typ": "JWT"
  },
  "payload": {
    "sub": "user-uuid-1234",
    "email": "researcher@university.edu",
    "org_id": "org-uuid-5678",
    "role": "admin",
    "permissions": ["compounds:read", "compounds:write", "compounds:delete"],
    "iat": 1700000000,
    "exp": 1700086400  // 24 hours
  },
  "signature": "..."
}
```

### RBAC Permissions

```python
# permissions.py
from enum import Enum

class Permission(Enum):
    # Compounds
    COMPOUNDS_READ = "compounds:read"
    COMPOUNDS_WRITE = "compounds:write"
    COMPOUNDS_DELETE = "compounds:delete"

    # Jobs
    JOBS_READ = "jobs:read"
    JOBS_CANCEL = "jobs:cancel"

    # Analytics
    ANALYTICS_VIEW = "analytics:view"
    ANALYTICS_EXPORT = "analytics:export"

    # Admin
    USERS_MANAGE = "users:manage"
    SETTINGS_MANAGE = "settings:manage"

ROLES = {
    'viewer': [
        Permission.COMPOUNDS_READ,
        Permission.JOBS_READ,
        Permission.ANALYTICS_VIEW
    ],
    'user': [
        Permission.COMPOUNDS_READ,
        Permission.COMPOUNDS_WRITE,
        Permission.JOBS_READ,
        Permission.JOBS_CANCEL,
        Permission.ANALYTICS_VIEW,
        Permission.ANALYTICS_EXPORT
    ],
    'admin': [  # All permissions
        *Permission
    ]
}

def check_permission(user: User, permission: Permission):
    """Check if user has permission."""
    if not user.is_active:
        raise PermissionError("User is inactive")

    user_permissions = ROLES.get(user.role, [])

    if permission not in user_permissions:
        raise PermissionError(f"User lacks permission: {permission.value}")
```

### API Key Authentication

```python
# For programmatic access (scripts, notebooks)

# api/auth.py
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def get_api_key(api_key: str = Security(api_key_header)):
    user = await get_user_by_api_key(api_key)
    if not user:
        raise HTTPException(403, "Invalid API key")
    return user

# Usage
@router.get('/compounds')
async def list_compounds(user: User = Depends(get_api_key)):
    # user authenticated via API key
    pass
```

### Rate Limiting

```python
# rate_limiter.py
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# Apply to routes
@router.post('/compounds')
@limiter.limit("10/minute")  # Max 10 submissions per minute
async def submit_compound(request: Request, ...):
    pass

@router.get('/compounds')
@limiter.limit("100/minute")  # Max 100 reads per minute
async def list_compounds(request: Request, ...):
    pass
```

### Data Encryption

```python
# At Rest: Encrypt sensitive fields in database
from cryptography.fernet import Fernet

cipher = Fernet(settings.ENCRYPTION_KEY)

def encrypt_field(value: str) -> str:
    return cipher.encrypt(value.encode()).decode()

def decrypt_field(encrypted: str) -> str:
    return cipher.decrypt(encrypted.encode()).decode()

# In Transit: HTTPS/TLS enforced at load balancer
# Internal: mTLS between services (optional)
```

---

## Monitoring & Observability

### Metrics (Prometheus)

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Job metrics
jobs_submitted = Counter('jobs_submitted_total', 'Total jobs submitted', ['job_type'])
jobs_completed = Counter('jobs_completed_total', 'Total jobs completed', ['job_type', 'status'])
job_duration = Histogram('job_duration_seconds', 'Job processing time', ['job_type'])

# Queue metrics
queue_depth = Gauge('queue_depth', 'Number of jobs in queue', ['queue_name'])

# API metrics
api_requests = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
api_latency = Histogram('api_latency_seconds', 'API latency', ['method', 'endpoint'])

# Database metrics
db_connections = Gauge('db_connections_active', 'Active database connections')
db_query_duration = Histogram('db_query_duration_seconds', 'Database query time', ['query_type'])

# Business metrics
compounds_processed_total = Counter('compounds_processed_total', 'Total compounds processed')
strong_imps_found = Counter('strong_imps_found_total', 'Total Strong IMPs identified')
```

### Grafana Dashboards

**System Overview Dashboard**:
- Total compounds processed (24h, 7d, 30d)
- Active users
- Queue depth (real-time)
- Worker utilization
- API request rate
- Error rate
- P95/P99 latency

**Job Processing Dashboard**:
- Jobs per hour
- Average job duration
- Job failure rate
- Queue wait time
- Worker auto-scaling history

**Business Metrics Dashboard**:
- O[Q/P/L]A score distribution
- Top compounds by score
- Most active users
- ChEMBL API usage
- PDB API usage

### Logging (ELK Stack)

```python
# logging_config.py
import logging
from pythonjsonlogger import jsonlogger

# Structured logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage
logger.info("Job submitted", extra={
    "job_id": job.id,
    "user_id": user.id,
    "job_type": job.job_type,
    "queue_position": 5
})

logger.error("Job failed", extra={
    "job_id": job.id,
    "error": str(exception),
    "retries": job.retries
}, exc_info=True)
```

### Alerting Rules

```yaml
# prometheus/alerts.yaml
groups:
- name: impulator_alerts
  rules:
  # Queue depth alert
  - alert: HighQueueDepth
    expr: queue_depth{queue_name="compounds"} > 100
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High queue depth detected"
      description: "Compounds queue has {{ $value }} pending jobs"

  # Worker down alert
  - alert: NoWorkersAvailable
    expr: up{job="impulator-worker"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "No workers available"
      description: "All workers are down"

  # High error rate
  - alert: HighErrorRate
    expr: rate(jobs_completed_total{status="failed"}[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High job failure rate"
      description: "{{ $value }}% of jobs are failing"

  # Database connection pool exhausted
  - alert: DatabaseConnectionPoolExhausted
    expr: db_connections_active >= 90
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Database connection pool near exhaustion"
      description: "{{ $value }} connections active (max 100)"

  # API latency
  - alert: HighAPILatency
    expr: histogram_quantile(0.95, rate(api_latency_seconds_bucket[5m])) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High API latency detected"
      description: "P95 latency is {{ $value }}s"
```

---

## Migration Strategy

### Phase 1: Proof of Concept (Months 1-2)

**Goal**: Build minimal viable production system with core features

**Tasks**:
1. Set up development environment
   - Docker Compose for local development
   - PostgreSQL, Redis, MinIO containers

2. Build API layer
   - FastAPI skeleton
   - Authentication (JWT)
   - Basic CRUD endpoints for compounds

3. Implement job queue
   - Celery setup
   - Single compound processing task
   - Job status tracking

4. Build minimal frontend
   - React app with Create React App
   - Login page
   - Compound submission form
   - Job status page

5. Deploy to staging
   - Single Kubernetes cluster
   - Manual deployment

**Deliverable**: Working prototype with 1-5 concurrent users

---

### Phase 2: Core Features (Months 3-4)

**Goal**: Complete feature parity with Streamlit prototype

**Tasks**:
1. Migrate all processing logic
   - ChEMBL integration
   - PDB integration
   - O[Q/P/L]A scoring
   - Assay interference detection

2. Build visualizations
   - D3.js efficiency planes
   - Three.js molecule viewer
   - Plotly charts

3. Implement batch processing
   - CSV upload
   - Parallel job processing
   - Progress tracking

4. Add data export
   - CSV export
   - PDF reports
   - SDF file generation

5. User management
   - Organizations
   - RBAC
   - User profiles

**Deliverable**: Feature-complete application for 10-20 users

---

### Phase 3: Scalability & Reliability (Months 5-6)

**Goal**: Handle 100-1000 concurrent users

**Tasks**:
1. Performance optimization
   - Database query optimization
   - API caching (Redis)
   - Connection pooling

2. Auto-scaling
   - Horizontal Pod Autoscaler for workers
   - API server auto-scaling
   - Load testing (Locust, k6)

3. Monitoring & observability
   - Prometheus + Grafana
   - ELK stack
   - Distributed tracing (Jaeger)

4. High availability
   - PostgreSQL replication
   - Redis Sentinel
   - Multi-region deployment (optional)

5. Security hardening
   - Penetration testing
   - OWASP compliance
   - Security audit

**Deliverable**: Production-ready system at scale

---

### Phase 4: Advanced Features (Months 7-9)

**Goal**: Enterprise features and optimizations

**Tasks**:
1. GraphQL API
2. Real-time collaboration (WebSockets)
3. Advanced analytics
4. Machine learning integration
5. Mobile app (React Native - optional)

**Deliverable**: Enterprise-grade application

---

## Cost Estimation

### Infrastructure Costs (AWS/GCP)

**Monthly costs for 100-1000 concurrent users:**

| Component | Specification | Monthly Cost (USD) |
|-----------|---------------|-------------------|
| **Compute** |  |  |
| API Servers (5x) | c6i.xlarge (4 vCPU, 8 GB) | $600 |
| Workers (avg 10x) | c6i.2xlarge (8 vCPU, 16 GB) | $1,800 |
| Frontend (3x) | t3.medium (2 vCPU, 4 GB) | $90 |
| **Database** |  |  |
| PostgreSQL RDS | db.r6g.xlarge (4 vCPU, 32 GB) | $500 |
| Redis ElastiCache | cache.r6g.large (2 vCPU, 13 GB) | $200 |
| **Storage** |  |  |
| S3 (10 TB) | Standard storage | $230 |
| EBS Volumes | 500 GB SSD | $50 |
| **Networking** |  |  |
| Load Balancer | Application LB | $40 |
| Data Transfer | 5 TB/month | $450 |
| **Monitoring** |  |  |
| CloudWatch | Logs + Metrics | $100 |
| **Total** |  | **$4,060/month** |

**Self-Hosted (On-Premise) Alternative**: ~$2,000/month (servers + bandwidth)

---

## Timeline & Roadmap

### Year 1 Roadmap

**Q1 2026** (Jan-Mar):
- [x] Complete architecture plan ✅
- [ ] Phase 1: POC (Months 1-2)
- [ ] Begin Phase 2: Core Features (Month 3)

**Q2 2026** (Apr-Jun):
- [ ] Complete Phase 2: Core Features
- [ ] Begin Phase 3: Scalability
- [ ] Beta launch (invite-only, 50 users)

**Q3 2026** (Jul-Sep):
- [ ] Complete Phase 3: Scalability
- [ ] Security audit & penetration testing
- [ ] Public launch (open registration)
- [ ] Begin Phase 4: Advanced Features

**Q4 2026** (Oct-Dec):
- [ ] Complete Phase 4: Advanced Features
- [ ] Enterprise features (SSO, SLA, support)
- [ ] Scale to 1000+ users
- [ ] API v2 with GraphQL

---

## Risk Analysis & Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **ChEMBL API downtime** | Medium | High | Cache results, fallback to local ChEMBL database |
| **PDB API rate limiting** | High | Medium | Request increase in limits, local PDB mirror |
| **Database performance degradation** | Medium | High | Read replicas, query optimization, caching |
| **Worker crashes during processing** | Medium | Medium | Job retries, checkpointing, idempotency |
| **Memory leaks in workers** | Medium | Medium | Worker restarts after N jobs, memory monitoring |
| **Storage costs exceed budget** | Low | Medium | Data lifecycle policies, compression |

### Business Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Low user adoption** | Medium | High | Marketing, partnerships, free tier |
| **Competitors emerge** | Medium | Medium | Continuous innovation, unique features |
| **Funding shortfall** | Low | High | Phased development, freemium model |
| **Data privacy concerns** | Low | Critical | GDPR compliance, data encryption, audit logs |
| **Vendor lock-in (cloud provider)** | Low | Medium | Use open-source components, multi-cloud design |

---

## Recommendations & Next Steps

### Immediate Actions (Week 1)

1. **Decision Points**:
   - [ ] Choose cloud provider (AWS, GCP, Azure, or self-hosted)
   - [ ] Select auth provider (Keycloak vs Auth0)
   - [ ] Decide on frontend framework (React confirmed)
   - [ ] Choose between REST-only or REST+GraphQL

2. **Team Formation**:
   - [ ] Hire/assign frontend developer (React + TypeScript)
   - [ ] Hire/assign backend developer (Python + FastAPI)
   - [ ] Hire/assign DevOps engineer (Kubernetes)
   - [ ] Assign project manager

3. **Setup**:
   - [ ] Create GitHub organization
   - [ ] Setup CI/CD pipeline (GitHub Actions)
   - [ ] Provision cloud accounts
   - [ ] Create project management board (Jira, Linear)

### Month 1 Priorities

1. **Infrastructure**:
   - Setup Kubernetes cluster
   - Deploy PostgreSQL, Redis, MinIO
   - Configure monitoring (Prometheus + Grafana)

2. **Backend**:
   - FastAPI skeleton with authentication
   - Database migrations (Alembic)
   - Celery task queue setup
   - Migrate one processing function (e.g., ChEMBL query)

3. **Frontend**:
   - Create React App setup
   - Authentication UI
   - Basic compound submission form

4. **Documentation**:
   - API documentation (OpenAPI/Swagger)
   - Developer onboarding guide
   - Architecture decision records (ADRs)

---

## Conclusion

This plan transforms IMPULATOR-3 from a single-user Streamlit prototype into a **production-ready, scalable application** capable of serving 100-1000 concurrent users.

### Key Achievements:

✅ **Scalability**: Horizontal scaling for API, workers, and database
✅ **Reliability**: Auto-scaling, fault tolerance, retries
✅ **Performance**: Asynchronous processing, caching, optimization
✅ **Security**: Authentication, authorization, encryption, audit logs
✅ **Observability**: Comprehensive monitoring, logging, tracing
✅ **User Experience**: Modern React frontend with real-time updates

### Success Metrics (6 months post-launch):

- **Active Users**: 500+ monthly active users
- **Compounds Processed**: 50,000+ total compounds
- **Uptime**: 99.9% availability
- **Performance**: <2s API latency (P95)
- **Queue Wait Time**: <30s average
- **User Satisfaction**: 4.5+ stars (user feedback)

---

**Document Version**: 1.0
**Last Updated**: November 19, 2025
**Status**: Ready for Review & Implementation

---

**We welcome feedback and suggestions! Please open a GitHub issue to discuss any aspect of this plan.**
