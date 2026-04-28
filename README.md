# Multi-Stage Job Ranking System

I built this project to understand how real recommendation and ranking systems are structured.

Instead of ranking every job directly, the system first retrieves a smaller set of candidates, then scores them with machine learning models, and finally applies simple product rules like freshness and diversity. I used SQLite so the whole project can run locally without extra infrastructure.
## What I learned

A good ranking system is not just a model. Retrieval quality matters, calibrated scores matter, and business rules can change what users actually see. I also learned that offline metrics can look strong even when a system still needs better diversity or better handling of cold-start items.
## Tradeoffs

I used SQLite and local vector retrieval instead of PostgreSQL/pgvector or FAISS because I wanted the project to be easy to run locally. The system is designed to show the ranking pipeline clearly, not to be optimized for large-scale production traffic.
