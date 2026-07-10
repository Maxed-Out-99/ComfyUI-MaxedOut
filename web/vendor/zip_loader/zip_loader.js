import { app } from "../../../../scripts/app.js";

function get_ext(filename) {
    const ext = filename.split(".").pop();
    if (!ext) {
        return "";
    }
    return ext.toLowerCase();
}

function get_common_top_folder(paths) {
    let common = null;
    for (const p of paths) {
        const parts = p.split("/").filter(Boolean);
        if (parts.length === 0) {
            return "";
        }
        const first = parts[0];
        if (common === null) {
            common = first;
        } else if (common !== first) {
            return "";
        }
    }
    return common || "";
}

app.registerExtension({
    name: "Comfy.ZipLoader",
    init() {
        // Use capture phase to intercept the event before ComfyUI's default handler (which likely listens on bubbling phase on document/body)
        document.addEventListener("drop", async (event) => {
            if (!event.dataTransfer || !event.dataTransfer.files || event.dataTransfer.files.length === 0) {
                return;
            }

            const files = Array.from(event.dataTransfer.files);
            let zipFiles = files.filter(f => get_ext(f.name) === "zip");
            let nonZipFiles = files.filter(f => get_ext(f.name) !== "zip");

            // If all dropped files are zips, we handle it and stop others
            if (zipFiles.length > 0 && nonZipFiles.length === 0) {
                event.preventDefault();
                event.stopPropagation();

                // Load JSZip if not already loaded
                if (!window.JSZip) {
                    try {
                        await import("./jszip.min.js");
                    } catch (e) {
                        console.error("Failed to load JSZip:", e);
                        alert("Failed to load JSZip library. Please check console.");
                        return;
                    }
                }

                let JSZip = window.JSZip;
                if (!JSZip) {
                    const module = await import("./jszip.min.js");
                    JSZip = module.default || module;
                }

                if (!JSZip) {
                    console.error("JSZip not found after import");
                    return;
                }

                let totalCount = 0;
                let workflowCount = 0;
                const workflowPaths = [];
                const workflowBlobs = [];

                for (const file of zipFiles) {
                    console.log("Processing zip file:", file.name);

                    try {
                        const zip = await JSZip.loadAsync(file);

                        let count = 0;
                        const promises = [];

                        zip.forEach((relativePath, zipEntry) => {
                            if (zipEntry.dir) return;
                            if (relativePath.startsWith("__MACOSX")) return;
                            if (relativePath.includes("/.")) return;
                            const ext = get_ext(relativePath);

                            const promise = zipEntry.async("blob").then(async (blob) => {
                                if (ext === "json") {
                                    workflowCount++;
                                    workflowPaths.push(relativePath);
                                    workflowBlobs.push(blob);
                                    return;
                                }

                                const targetPath = "workflows/" + relativePath;
                                const url = `/api/userdata/${encodeURIComponent(targetPath)}?overwrite=true`;

                                const res = await fetch(url, {
                                    method: "POST",
                                    body: blob
                                });

                                if (res.ok) {
                                    count++;
                                    totalCount++;
                                } else {
                                    console.error("Failed to upload:", relativePath, res.statusText);
                                }
                            });
                            promises.push(promise);
                        });

                        await Promise.all(promises);

                    } catch (err) {
                        console.error("Error processing zip:", err);
                        alert("Error processing zip file: " + err.message);
                    }
                }

                let loadedSingleInMemory = false;
                if (workflowCount === 1 && typeof app.handleFile === "function") {
                    // Mirror vanilla ComfyUI's own json-drop behavior: load straight into the
                    // graph via app.handleFile instead of writing to disk and forcing a reload.
                    try {
                        const relativePath = workflowPaths[0];
                        const blob = workflowBlobs[0];
                        const filename = relativePath.split("/").pop();
                        const jsonFile = new File([blob], filename, { type: "application/json" });
                        await app.handleFile(jsonFile);
                        loadedSingleInMemory = true;
                    } catch (err) {
                        // app.handleFile is an internal API; fall back to the disk-upload
                        // path below if a future ComfyUI build changes/removes it.
                        console.error("app.handleFile failed, falling back to workflow upload:", err);
                    }
                }

                if (!loadedSingleInMemory && workflowCount > 0) {
                    await Promise.all(workflowPaths.map(async (relativePath, i) => {
                        const targetPath = "workflows/" + relativePath;
                        const url = `/api/userdata/${encodeURIComponent(targetPath)}?overwrite=true`;

                        const res = await fetch(url, {
                            method: "POST",
                            body: workflowBlobs[i]
                        });

                        if (res.ok) {
                            totalCount++;
                        } else {
                            console.error("Failed to upload:", relativePath, res.statusText);
                        }
                    }));

                    const sortedWorkflows = workflowPaths.slice().sort((a, b) => a.localeCompare(b));
                    const commonFolder = get_common_top_folder(sortedWorkflows) || "(Root)";

                    const modal = document.createElement("div");
                    Object.assign(modal.style, {
                        position: "fixed",
                        top: "0",
                        left: "0",
                        width: "100%",
                        height: "100%",
                        backgroundColor: "rgba(0,0,0,0.8)",
                        display: "flex",
                        justifyContent: "center",
                        alignItems: "center",
                        zIndex: "10000",
                        fontFamily: "sans-serif"
                    });

                    const dialog = document.createElement("div");
                    Object.assign(dialog.style, {
                        backgroundColor: "#222",
                        color: "white",
                        padding: "25px",
                        borderRadius: "10px",
                        width: "450px",
                        border: "1px solid #444",
                        boxShadow: "0 0 20px rgba(0,0,0,0.5)"
                    });

                    dialog.innerHTML = `
                        <h2 style="margin-top:0; color:#44cf7e;">Workflows Imported</h2>
                        <p><strong>Folder:</strong> <span style="color:#00bdff;">${commonFolder}</span></p>
                        <div style="background:#111; padding:10px; border-radius:5px; max-height:200px; overflow-y:auto; margin:15px 0;">
                            <ul id="workflowList" style="margin:0; padding-left:20px; font-size:14px; line-height:1.6;"></ul>
                        </div>
                        <p style="font-size:13px; color:#aaa;">
                            These will appear in the <strong>Workflows</strong> side panel (Press <strong>W</strong>) after a reload.
                        </p>
                        <div style="display:flex; gap:10px; margin-top:20px;">
                            <button id="reloadBtn" style="flex:1; padding:10px; background:#44cf7e; border:none; color:black; font-weight:bold; border-radius:4px; cursor:pointer;">Reload Now</button>
                            <button id="closeBtn" style="flex:1; padding:10px; background:#444; border:none; color:white; border-radius:4px; cursor:pointer;">Later</button>
                        </div>
                    `;
                    const workflowList = dialog.querySelector("#workflowList");
                    for (const p of sortedWorkflows) {
                        const li = document.createElement("li");
                        li.textContent = p.split("/").pop();
                        workflowList.appendChild(li);
                    }

                    modal.appendChild(dialog);
                    document.body.appendChild(modal);

                    dialog.querySelector("#reloadBtn").onclick = () => window.location.reload();
                    dialog.querySelector("#closeBtn").onclick = () => document.body.removeChild(modal);
                }
            }
        }, true); // Capture = true

        // We also need to prevent default dragover to allow drop
        document.addEventListener("dragover", (event) => {
            if (event.dataTransfer && event.dataTransfer.types && event.dataTransfer.types.includes("Files")) {
                // event.preventDefault(); // This is needed to allow drop
            }
        }, true);
    }
});
