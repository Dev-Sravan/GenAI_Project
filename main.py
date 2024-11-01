if __name__ == "__main__":
    import uvicorn
    uvicorn.run("router.router:router", reload=True)