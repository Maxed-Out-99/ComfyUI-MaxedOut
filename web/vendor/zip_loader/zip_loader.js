import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

function get_ext(filename) {
    const ext = filename.split(".").pop();
    if (!ext) {
        return "";
    }
    return ext.toLowerCase();
}

/** Normalize zip entry paths: forward slashes, no leading ./ */
function normalize_zip_path(relativePath) {
    let p = String(relativePath || "").replace(/\\/g, "/");
    while (p.startsWith("./")) {
        p = p.slice(2);
    }
    return p.replace(/\/+/g, "/");
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

function should_skip_entry(relativePath) {
    if (!relativePath || relativePath.endsWith("/")) return true;
    if (relativePath.startsWith("__MACOSX/") || relativePath.includes("/__MACOSX/")) return true;
    const parts = relativePath.split("/");
    if (parts.some((part) => part.startsWith("."))) return true;
    return false;
}

async function upload_userdata(targetPath, blob) {
    if (api && typeof api.storeUserData === "function") {
        const res = await api.storeUserData(targetPath, blob, {
            overwrite: true,
            stringify: false,
            throwOnError: false,
            full_info: false
        });
        return res;
    }
    return api.fetchApi(`/userdata/${encodeURIComponent(targetPath)}?overwrite=true`, {
        method: "POST",
        body: blob
    });
}

async function refresh_workflows_sidebar() {
    try {
        const pinia = app?.vueApp?.config?.globalProperties?.$pinia;
        const store = pinia?._s?.get?.("workflow");
        if (store && typeof store.syncWorkflows === "function") {
            await store.syncWorkflows();
            return true;
        }
    } catch (err) {
        console.warn("zip_loader: syncWorkflows via vueApp failed", err);
    }
    try {
        const stores = window.__PINIA__?._s;
        const store = stores?.get?.("workflow");
        if (store && typeof store.syncWorkflows === "function") {
            await store.syncWorkflows();
            return true;
        }
    } catch (err) {
        console.warn("zip_loader: syncWorkflows via __PINIA__ failed", err);
    }
    return false;
}

app.registerExtension({
    name: "Comfy.ZipLoader",
    init() {
        document.addEventListener("drop", async (event) => {
            if (!event.dataTransfer || !event.dataTransfer.files || event.dataTransfer.files.length === 0) {
                return;
            }

            const files = Array.from(event.dataTransfer.files);
            let zipFiles = files.filter(f => get_ext(f.name) === "zip");
            let nonZipFiles = files.filter(f => get_ext(f.name) !== "zip");

            if (zipFiles.length > 0 && nonZipFiles.length === 0) {
                event.preventDefault();
                event.stopPropagation();

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
                const failedUploads = [];

                for (const file of zipFiles) {
                    console.log("Processing zip file:", file.name);

                    try {
                        const zip = await JSZip.loadAsync(file);
                        const promises = [];

                        zip.forEach((relativePath, zipEntry) => {
                            if (zipEntry.dir) return;
                            const normPath = normalize_zip_path(relativePath);
                            if (should_skip_entry(normPath)) return;
                            const ext = get_ext(normPath);

                            const promise = zipEntry.async("blob").then(async (blob) => {
                                if (ext === "json") {
                                    workflowCount++;
                                    workflowPaths.push(normPath);
                                    workflowBlobs.push(blob);
                                    return;
                                }

                                const targetPath = "workflows/" + normPath;
                                const res = await upload_userdata(targetPath, blob);

                                if (res.ok) {
                                    totalCount++;
                                } else {
                                    console.error("Failed to upload:", normPath, res.status, res.statusText);
                                    failedUploads.push(normPath);
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
                    try {
                        const relativePath = workflowPaths[0];
                        const blob = workflowBlobs[0];
                        const filename = relativePath.split("/").pop();
                        const jsonFile = new File([blob], filename, { type: "application/json" });
                        await app.handleFile(jsonFile);
                        loadedSingleInMemory = true;
                    } catch (err) {
                        console.error("app.handleFile failed, falling back to workflow upload:", err);
                    }
                }

                if (!loadedSingleInMemory && workflowCount > 0) {
                    const importedWorkflowPaths = [];
                    await Promise.all(workflowPaths.map(async (relativePath, i) => {
                        const targetPath = "workflows/" + relativePath;
                        const res = await upload_userdata(targetPath, workflowBlobs[i]);

                        if (res.ok) {
                            totalCount++;
                            importedWorkflowPaths.push(relativePath);
                        } else {
                            console.error("Failed to upload:", relativePath, res.status, res.statusText);
                            failedUploads.push(relativePath);
                        }
                    }));

                    if (importedWorkflowPaths.length === 0) {
                        alert("Workflow import failed. No files were saved. Please check the browser console for details.");
                        return;
                    }

                    const synced = await refresh_workflows_sidebar();
                    const sortedWorkflows = importedWorkflowPaths.slice().sort((a, b) => a.localeCompare(b));
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

                    const whereHint = synced
                        ? `They should already be under <strong>Workflows</strong> (press <strong>W</strong>). Look in folder <span style="color:#00bdff;">${commonFolder}</span>.`
                        : `Open <strong>Workflows</strong> (press <strong>W</strong>) after reload. Look in folder <span style="color:#00bdff;">${commonFolder}</span>.`;

                    dialog.innerHTML = `
                        <h2 style="margin-top:0; color:#44cf7e;">Workflows Imported</h2>
                        <p><strong>Folder:</strong> <span style="color:#00bdff;">${commonFolder}</span></p>
                        <div style="background:#111; padding:10px; border-radius:5px; max-height:200px; overflow-y:auto; margin:15px 0;">
                            <ul id="workflowList" style="margin:0; padding-left:20px; font-size:14px; line-height:1.6;"></ul>
                        </div>
                        <p style="font-size:13px; color:#aaa;">
                            ${whereHint}
                        </p>
                        <div style="display:flex; gap:10px; margin-top:20px;">
                            <button id="reloadBtn" style="flex:1; padding:10px; background:#44cf7e; border:none; color:black; font-weight:bold; border-radius:4px; cursor:pointer;">${synced ? "Done" : "Reload Now"}</button>
                            <button id="closeBtn" style="flex:1; padding:10px; background:#444; border:none; color:white; border-radius:4px; cursor:pointer;">${synced ? "Close" : "Later"}</button>
                        </div>
                    `;
                    const workflowList = dialog.querySelector("#workflowList");
                    for (const p of sortedWorkflows) {
                        const li = document.createElement("li");
                        li.textContent = p;
                        workflowList.appendChild(li);
                    }

                    modal.appendChild(dialog);
                    document.body.appendChild(modal);

                    if (failedUploads.length > 0) {
                        const warning = document.createElement("p");
                        warning.style.color = "#ffb84d";
                        warning.textContent = `${failedUploads.length} file(s) failed to import. Check the browser console for details.`;
                        dialog.insertBefore(warning, dialog.querySelector("div:last-child"));
                    }

                    dialog.querySelector("#reloadBtn").onclick = async () => {
                        if (synced) {
                            document.body.removeChild(modal);
                            return;
                        }
                        const ok = await refresh_workflows_sidebar();
                        if (ok) {
                            document.body.removeChild(modal);
                            return;
                        }
                        window.location.reload();
                    };
                    dialog.querySelector("#closeBtn").onclick = () => document.body.removeChild(modal);
                }
            }
        }, true);

        document.addEventListener("dragover", (event) => {
            if (event.dataTransfer && event.dataTransfer.types && event.dataTransfer.types.includes("Files")) {
                event.preventDefault();
            }
        }, true);
    }
});