export AZCOPY_AUTO_LOGIN_TYPE=AZCLI

success=true

for i in $(seq 25 40); do
    idx=$(printf "%03d" "$i")

    azcopy copy --recursive \
        "/Data/Ego10k/factory_${idx}" \
        "https://azsussc.blob.core.windows.net/v-wangxiaofa/robot_dataset/ego_centric_10k/"

    if [ $? -ne 0 ]; then
        success=false
        echo "Upload failed: factory_${idx}"
        break
    fi
done

if [ "$success" = true ]; then
    echo "All uploads succeeded, deleting local directories..."
    rm -rf /Data/Ego10k/factory_{025..040}
fi